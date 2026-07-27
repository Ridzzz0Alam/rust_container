//! A minimal Linux container runtime, in Rust.
//!
//! A faithful port of the ~500-line C container from
//! "Linux Containers in 500 Lines of Code" by Lizzie Dixon.
//!
//! It isolates a process using Linux namespaces, remaps UIDs/GIDs with a user
//! namespace, swaps the root filesystem with pivot_root, drops dangerous
//! capabilities, and installs a seccomp syscall filter before exec-ing the
//! target command.
//!
//! Build:  cargo build --release
//! Run:    sudo ./target/release/contained -m ./rootfs -u 0 -c /bin/sh

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fs::OpenOptions;
use std::io::Write;
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use caps::{CapSet, Capability};
use libseccomp::{ScmpAction, ScmpArgCompare, ScmpCompareOp, ScmpFilterContext, ScmpSyscall};
use nix::mount::{mount, umount2, MntFlags, MsFlags};
use nix::sched::{clone, unshare, CloneCb, CloneFlags};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{chdir, execve, pivot_root, setgroups, setresgid, setresuid, sethostname, Gid, Pid, Uid};

/// Shorthand result type: any error converts into a boxed trait object.
type R = Result<(), Box<dyn Error>>;

const USERNS_OFFSET: i32 = 10_000;
const USERNS_COUNT: i32 = 2_000;
const STACK_SIZE: usize = 1024 * 1024;

/// Everything the cloned child needs to set itself up.
struct ChildConfig {
    uid: u32,
    /// Child's end of the socketpair (used to coordinate uid_map with the parent).
    fd: RawFd,
    hostname: String,
    /// Command + its arguments (argv[0] is the program to exec).
    argv: Vec<CString>,
    mount_dir: PathBuf,
}

// ----------------------------------------------------------------------------
// Tiny helpers for the 4-byte int handshake over the socketpair (mirrors the
// C code, which reads/writes sizeof(int) raw bytes).
// ----------------------------------------------------------------------------

fn write_int(fd: RawFd, val: i32) -> R {
    let bytes = val.to_ne_bytes();
    let n = unsafe { libc::write(fd, bytes.as_ptr() as *const libc::c_void, 4) };
    if n != 4 {
        return Err(format!("couldn't write: {}", std::io::Error::last_os_error()).into());
    }
    Ok(())
}

fn read_int(fd: RawFd) -> Result<i32, Box<dyn Error>> {
    let mut bytes = [0u8; 4];
    let n = unsafe { libc::read(fd, bytes.as_mut_ptr() as *mut libc::c_void, 4) };
    if n != 4 {
        return Err(format!("couldn't read: {}", std::io::Error::last_os_error()).into());
    }
    Ok(i32::from_ne_bytes(bytes))
}

fn cstr_array_to_string(ptr: *const libc::c_char) -> String {
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// mkdtemp(3): create a uniquely-named temp directory from a template.
fn mkdtemp(template: &str) -> Result<PathBuf, Box<dyn Error>> {
    let mut buf: Vec<u8> = template.as_bytes().to_vec();
    buf.push(0); // NUL terminator for C
    let ret = unsafe { libc::mkdtemp(buf.as_mut_ptr() as *mut libc::c_char) };
    if ret.is_null() {
        return Err(format!("mkdtemp failed: {}", std::io::Error::last_os_error()).into());
    }
    let end = buf.iter().position(|&c| c == 0).unwrap();
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(&buf[..end])))
}

// ----------------------------------------------------------------------------
// capabilities: drop dangerous capabilities from the bounding + inheritable sets
// ----------------------------------------------------------------------------

fn capabilities() -> R {
    eprint!("=> dropping capabilities...");
    let drop_caps = [
        Capability::CAP_AUDIT_CONTROL,
        Capability::CAP_AUDIT_READ,
        Capability::CAP_AUDIT_WRITE,
        Capability::CAP_BLOCK_SUSPEND,
        Capability::CAP_DAC_READ_SEARCH,
        Capability::CAP_FSETID,
        Capability::CAP_IPC_LOCK,
        Capability::CAP_MAC_ADMIN,
        Capability::CAP_MAC_OVERRIDE,
        Capability::CAP_MKNOD,
        Capability::CAP_SETFCAP,
        Capability::CAP_SYSLOG,
        Capability::CAP_SYS_ADMIN,
        Capability::CAP_SYS_BOOT,
        Capability::CAP_SYS_MODULE,
        Capability::CAP_SYS_NICE,
        Capability::CAP_SYS_RAWIO,
        Capability::CAP_SYS_RESOURCE,
        Capability::CAP_SYS_TIME,
        Capability::CAP_WAKE_ALARM,
    ];

    eprint!("bounding...");
    for cap in drop_caps.iter() {
        caps::drop(None, CapSet::Bounding, *cap)
            .map_err(|e| format!("prctl failed: {}", e))?;
    }

    eprint!("inheritable...");
    for cap in drop_caps.iter() {
        caps::drop(None, CapSet::Inheritable, *cap)
            .map_err(|e| format!("failed: {}", e))?;
    }

    eprintln!("done.");
    Ok(())
}

// ----------------------------------------------------------------------------
// mounts: the "mount dance" -- bind mount the rootfs, pivot_root into it, and
// detach the old root so the container cannot see the host filesystem.
// ----------------------------------------------------------------------------

fn mounts(config: &ChildConfig) -> R {
    eprint!("=> remounting everything with MS_PRIVATE...");
    mount(
        None::<&Path>,
        Path::new("/"),
        None::<&Path>,
        MsFlags::MS_REC | MsFlags::MS_PRIVATE,
        None::<&Path>,
    )
    .map_err(|e| format!("failed! {}", e))?;
    eprintln!("remounted.");

    eprint!("=> making a temp directory and a bind mount there...");
    let mount_dir = mkdtemp("/tmp/tmp.XXXXXX")?;

    mount(
        Some(config.mount_dir.as_path()),
        mount_dir.as_path(),
        None::<&Path>,
        MsFlags::MS_BIND | MsFlags::MS_PRIVATE,
        None::<&Path>,
    )
    .map_err(|_| "bind mount failed!")?;

    // Create the directory the old root will be moved to, inside the new root.
    let inner_template = format!("{}/oldroot.XXXXXX", mount_dir.display());
    let inner_mount_dir = mkdtemp(&inner_template)?;
    eprintln!("done.");

    eprint!("=> pivoting root...");
    pivot_root(mount_dir.as_path(), inner_mount_dir.as_path())
        .map_err(|_| "failed!")?;
    eprintln!("done.");

    // After pivot_root, the old root lives at "/<basename of inner_mount_dir>".
    let old_root_name = inner_mount_dir
        .file_name()
        .ok_or("bad inner mount dir")?;
    let mut old_root = PathBuf::from("/");
    old_root.push(old_root_name);

    eprint!("=> unmounting {}...", old_root.display());
    chdir("/").map_err(|e| format!("chdir failed! {}", e))?;
    umount2(old_root.as_path(), MntFlags::MNT_DETACH)
        .map_err(|e| format!("umount failed! {}", e))?;
    std::fs::remove_dir(&old_root).map_err(|e| format!("rmdir failed! {}", e))?;
    eprintln!("done.");
    Ok(())
}

// ----------------------------------------------------------------------------
// syscalls: install a seccomp filter that EPERMs a handful of dangerous calls.
// ----------------------------------------------------------------------------

fn syscalls() -> R {
    eprint!("=> filtering syscalls...");
    let eperm = ScmpAction::Errno(libc::EPERM);
    let s_isuid = libc::S_ISUID as u64;
    let s_isgid = libc::S_ISGID as u64;
    let clone_newuser = libc::CLONE_NEWUSER as u64;
    let tiocsti = libc::TIOCSTI as u64;

    let mut ctx = ScmpFilterContext::new_filter(ScmpAction::Allow)?;

    // Block setuid/setgid bits via chmod family (privilege escalation).
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("chmod")?,
        &[ScmpArgCompare::new(1, ScmpCompareOp::MaskedEqual(s_isuid), s_isuid)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("chmod")?,
        &[ScmpArgCompare::new(1, ScmpCompareOp::MaskedEqual(s_isgid), s_isgid)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("fchmod")?,
        &[ScmpArgCompare::new(1, ScmpCompareOp::MaskedEqual(s_isuid), s_isuid)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("fchmod")?,
        &[ScmpArgCompare::new(1, ScmpCompareOp::MaskedEqual(s_isgid), s_isgid)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("fchmodat")?,
        &[ScmpArgCompare::new(2, ScmpCompareOp::MaskedEqual(s_isuid), s_isuid)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("fchmodat")?,
        &[ScmpArgCompare::new(2, ScmpCompareOp::MaskedEqual(s_isgid), s_isgid)])?;

    // Block creating nested user namespaces from inside the container.
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("unshare")?,
        &[ScmpArgCompare::new(0, ScmpCompareOp::MaskedEqual(clone_newuser), clone_newuser)])?;
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("clone")?,
        &[ScmpArgCompare::new(0, ScmpCompareOp::MaskedEqual(clone_newuser), clone_newuser)])?;

    // Block TIOCSTI ioctl (terminal input injection into the parent tty).
    ctx.add_rule_conditional(eperm, ScmpSyscall::from_name("ioctl")?,
        &[ScmpArgCompare::new(1, ScmpCompareOp::MaskedEqual(tiocsti), tiocsti)])?;

    // Block outright: keyring (not namespaced), ptrace, NUMA memory policy,
    // userfaultfd and perf (kernel-exploit primitives).
    for name in [
        "keyctl", "add_key", "request_key", "ptrace", "mbind", "migrate_pages",
        "move_pages", "set_mempolicy", "userfaultfd", "perf_event_open",
    ] {
        ctx.add_rule(eperm, ScmpSyscall::from_name(name)?)?;
    }

    // Don't force NO_NEW_PRIVS (the C code sets SCMP_FLTATR_CTL_NNP = 0).
    ctx.set_ctl_nnp(false)?;
    ctx.load()?;

    eprintln!("done.");
    Ok(())
}

// ----------------------------------------------------------------------------
// user namespace: unshare CLONE_NEWUSER, coordinate the uid/gid map with the
// parent over the socketpair, then switch to the target uid/gid.
// ----------------------------------------------------------------------------

fn userns(config: &ChildConfig) -> R {
    eprint!("=> trying a user namespace...");
    let has_userns: i32 = match unshare(CloneFlags::CLONE_NEWUSER) {
        Ok(()) => 1,
        Err(_) => 0,
    };
    write_int(config.fd, has_userns)?;
    let result = read_int(config.fd)?;
    if result != 0 {
        return Err("parent failed to set up uid_map".into());
    }
    if has_userns == 1 {
        eprintln!("done.");
    } else {
        eprintln!("unsupported? continuing.");
    }

    eprint!("=> switching to uid {} / gid {}...", config.uid, config.uid);
    let uid = Uid::from_raw(config.uid);
    let gid = Gid::from_raw(config.uid);
    setgroups(&[gid]).map_err(|e| format!("{}", e))?;
    setresgid(gid, gid, gid).map_err(|e| format!("{}", e))?;
    setresuid(uid, uid, uid).map_err(|e| format!("{}", e))?;
    eprintln!("done.");
    Ok(())
}

/// The parent side of the user-namespace handshake: read whether the child made
/// a userns, write /proc/<pid>/{uid_map,gid_map}, then signal completion.
fn handle_child_uid_map(child_pid: Pid, fd: RawFd) -> R {
    let has_userns = read_int(fd)?;
    if has_userns != 0 {
        for file in ["uid_map", "gid_map"] {
            let path = format!("/proc/{}/{}", child_pid.as_raw(), file);
            eprint!("writing {}...", path);
            let mut f = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| format!("open failed: {}", e))?;
            // The proc uid_map/gid_map interface requires the entire mapping to
            // be delivered in a SINGLE write() call. Format the line first, then
            // write it all at once (do NOT use write!(), which may fragment the
            // output into several write() syscalls and cause EINVAL).
            let line = format!("0 {} {}\n", USERNS_OFFSET, USERNS_COUNT);
            f.write_all(line.as_bytes())
                .map_err(|e| format!("write failed: {}", e))?;
        }
    }
    // Signal the child that the maps are ready.
    write_int(fd, 0)?;
    Ok(())
}

// ----------------------------------------------------------------------------
// child entry point: runs in the new namespaces, then execs the command.
// ----------------------------------------------------------------------------

fn child(config: &ChildConfig) -> isize {
    let setup = (|| -> R {
        sethostname(&config.hostname)?;
        mounts(config)?;
        userns(config)?;
        capabilities()?;
        syscalls()?;
        Ok(())
    })();

    if let Err(e) = setup {
        eprintln!("child setup failed: {}", e);
        unsafe { libc::close(config.fd) };
        return -1;
    }

    unsafe { libc::close(config.fd) };

    let env: Vec<CString> = Vec::new();
    match execve(&config.argv[0], &config.argv, &env) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("execve failed! {}.", e);
            -1
        }
    }
}

// ----------------------------------------------------------------------------
// choose_hostname: a whimsical tarot-card hostname, like the original.
// ----------------------------------------------------------------------------

fn choose_hostname() -> String {
    let suits = ["swords", "wands", "pentacles", "cups"];
    let minor = [
        "ace", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "page", "knight", "queen", "king",
    ];
    let major = [
        "fool", "magician", "high-priestess", "empress", "emperor", "hierophant",
        "lovers", "chariot", "strength", "hermit", "wheel", "justice",
        "hanged-man", "death", "temperance", "devil", "tower", "star", "moon",
        "sun", "judgment", "world",
    ];

    let mut now: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };

    let ix = (now.tv_nsec as u64) % 78;
    let major_len = major.len() as u64;
    if ix < major_len {
        format!("{:05x}-{}", now.tv_sec, major[ix as usize])
    } else {
        let ix = ix - major_len;
        let minor_len = minor.len() as u64;
        format!(
            "{:05x}c-{}-of-{}",
            now.tv_sec,
            minor[(ix % minor_len) as usize],
            suits[(ix / minor_len) as usize]
        )
    }
}

// ----------------------------------------------------------------------------
// argument parsing
// ----------------------------------------------------------------------------

fn usage(prog: &str) -> ! {
    eprintln!("Usage: {} -u <uid> -m <rootfs_dir> -c <cmd> [args...]", prog);
    std::process::exit(1);
}

struct Args {
    uid: u32,
    mount_dir: PathBuf,
    argv: Vec<CString>,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let prog = raw.get(0).cloned().unwrap_or_else(|| "contained".into());

    let mut uid: u32 = 0;
    let mut mount_dir: Option<String> = None;
    let mut argv: Vec<CString> = Vec::new();

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "-u" => {
                i += 1;
                let v = raw.get(i).unwrap_or_else(|| usage(&prog));
                uid = v.parse().unwrap_or_else(|_| {
                    eprintln!("badly-formatted uid: {}", v);
                    usage(&prog)
                });
            }
            "-m" => {
                i += 1;
                mount_dir = Some(raw.get(i).unwrap_or_else(|| usage(&prog)).clone());
            }
            "-c" => {
                i += 1;
                for a in &raw[i..] {
                    argv.push(CString::new(a.as_str()).expect("NUL in argument"));
                }
                break;
            }
            _ => usage(&prog),
        }
        i += 1;
    }

    let mount_dir = mount_dir.unwrap_or_else(|| usage(&prog));
    if argv.is_empty() {
        usage(&prog);
    }
    Args {
        uid,
        mount_dir: PathBuf::from(mount_dir),
        argv,
    }
}

fn main() {
    let args = parse_args();

    // --- validate kernel / arch (mirrors the C uname check) ---
    eprint!("=> validating Linux version...");
    let mut u: libc::utsname = unsafe { std::mem::zeroed() };
    if unsafe { libc::uname(&mut u) } != 0 {
        eprintln!("failed: {}", std::io::Error::last_os_error());
        std::process::exit(1);
    }
    let release = cstr_array_to_string(u.release.as_ptr());
    let machine = cstr_array_to_string(u.machine.as_ptr());

    let major: u32 = release
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            eprintln!("weird release format: {}", release);
            std::process::exit(1);
        });
    if major < 4 {
        eprintln!("kernel too old: {}", release);
        std::process::exit(1);
    }
    if machine != "x86_64" {
        eprintln!("expected x86_64: {}", machine);
        std::process::exit(1);
    }
    eprintln!("{} on {}.", release, machine);

    let hostname = choose_hostname();

    // --- socketpair for the uid_map handshake ---
    let mut sockets = [0 as RawFd; 2];
    if unsafe { libc::socketpair(libc::AF_LOCAL, libc::SOCK_SEQPACKET, 0, sockets.as_mut_ptr()) } != 0 {
        eprintln!("socketpair failed: {}", std::io::Error::last_os_error());
        std::process::exit(1);
    }
    // Parent's end must not leak into the child on exec.
    unsafe { libc::fcntl(sockets[0], libc::F_SETFD, libc::FD_CLOEXEC) };

    let config = ChildConfig {
        uid: args.uid,
        fd: sockets[1],
        hostname,
        argv: args.argv,
        mount_dir: args.mount_dir,
    };

    // NOTE: cgroup resource limits (see `resources()` below) are intentionally
    // NOT applied here -- the original targets cgroups v1, but modern systems
    // (and WSL2) use cgroups v2. Enable/port them if you're on cgroups v1.

    let flags = CloneFlags::CLONE_NEWNS
        | CloneFlags::CLONE_NEWCGROUP
        | CloneFlags::CLONE_NEWPID
        | CloneFlags::CLONE_NEWIPC
        | CloneFlags::CLONE_NEWNET
        | CloneFlags::CLONE_NEWUTS;

    let mut stack = vec![0u8; STACK_SIZE];
    let cb: CloneCb = Box::new(|| child(&config) as isize);

    // SAFETY: no CLONE_VM, so the child gets a copy-on-write copy of memory
    // (fork-like). The closure only touches its own copy of `config`.
    let child_pid = match unsafe { clone(cb, &mut stack, flags, Some(libc::SIGCHLD)) } {
        Ok(pid) => pid,
        Err(e) => {
            eprintln!("=> clone failed! {}", e);
            std::process::exit(1);
        }
    };

    // Parent no longer needs the child's end of the socket.
    unsafe { libc::close(sockets[1]) };

    let mut err = 0;
    if let Err(e) = handle_child_uid_map(child_pid, sockets[0]) {
        eprintln!("uid map handling failed: {}", e);
        let _ = nix::sys::signal::kill(child_pid, nix::sys::signal::Signal::SIGKILL);
        err = 1;
    }

    match waitpid(child_pid, None) {
        Ok(WaitStatus::Exited(_, code)) => err |= code,
        Ok(_) => {}
        Err(e) => {
            eprintln!("waitpid failed: {}", e);
            err = 1;
        }
    }

    unsafe { libc::close(sockets[0]) };
    std::process::exit(err);
}

// ----------------------------------------------------------------------------
// cgroups v1 resource limits -- ported for completeness but NOT wired in,
// exactly like the original (its call site in main was commented out because
// modern kernels use cgroups v2). Kept here for reference.
// ----------------------------------------------------------------------------

#[allow(dead_code)]
mod resources {
    use std::fs;
    use std::io::Write;

    const MEMORY: &str = "1073741824";
    const SHARES: &str = "256";
    const PIDS: &str = "64";
    const FD_COUNT: u64 = 64;

    struct Setting {
        name: &'static str,
        value: &'static str,
    }
    struct Control {
        control: &'static str,
        settings: Vec<Setting>,
    }

    fn controls() -> Vec<Control> {
        vec![
            Control {
                control: "memory",
                settings: vec![
                    Setting { name: "memory.limit_in_bytes", value: MEMORY },
                    Setting { name: "memory.kmem.limit_in_bytes", value: MEMORY },
                    Setting { name: "tasks", value: "0" },
                ],
            },
            Control {
                control: "cpu",
                settings: vec![
                    Setting { name: "cpu.shares", value: SHARES },
                    Setting { name: "tasks", value: "0" },
                ],
            },
            Control {
                control: "pids",
                settings: vec![
                    Setting { name: "pids.max", value: PIDS },
                    Setting { name: "tasks", value: "0" },
                ],
            },
            Control {
                control: "blkio",
                settings: vec![
                    Setting { name: "blkio.weight", value: PIDS },
                    Setting { name: "tasks", value: "0" },
                ],
            },
        ]
    }

    pub fn set_resources(hostname: &str) -> std::io::Result<()> {
        for cgrp in controls() {
            let dir = format!("/sys/fs/cgroup/{}/{}", cgrp.control, hostname);
            fs::create_dir_all(&dir)?;
            for s in &cgrp.settings {
                let path = format!("{}/{}", dir, s.name);
                let mut f = fs::OpenOptions::new().write(true).open(&path)?;
                f.write_all(s.value.as_bytes())?;
            }
        }
        let lim = libc::rlimit { rlim_cur: FD_COUNT, rlim_max: FD_COUNT };
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lim) };
        Ok(())
    }

    pub fn free_resources(hostname: &str) -> std::io::Result<()> {
        for cgrp in controls() {
            let dir = format!("/sys/fs/cgroup/{}/{}", cgrp.control, hostname);
            let task = format!("/sys/fs/cgroup/{}/tasks", cgrp.control);
            let mut f = fs::OpenOptions::new().write(true).open(&task)?;
            f.write_all(b"0")?;
            fs::remove_dir(&dir)?;
        }
        Ok(())
    }
}