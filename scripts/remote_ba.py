"""SSH helper: credentials from env MY_SERVER_INFOR ("ip/username/password").
Mật khẩu chỉ đọc từ biến môi trường, không bao giờ ghi ra file/log.
Cần: pip install paramiko

Usage:
  python scripts/remote_ba.py run "<command>"          # runs inside /home/darkweb/BA
  python scripts/remote_ba.py put <local> <remote_rel> # upload into /home/darkweb/BA/<remote_rel>
  python scripts/remote_ba.py get <remote_rel> <local> # download from /home/darkweb/BA/<remote_rel>

Every remote command is prefixed with `cd /home/darkweb/BA` and an environment
that keeps HOME/caches/tmp inside that folder. Paths are refused if they
resolve outside BASE.
"""
import os
import posixpath
import sys

import paramiko

BASE = "/home/darkweb/BA"
ENV = (
    f"export HOME={BASE}/.home XDG_CACHE_HOME={BASE}/.home/.cache XDG_CONFIG_HOME={BASE}/.home/.config "
    f"TMPDIR={BASE}/.tmp MPLCONFIGDIR={BASE}/.home/.mpl UV_CACHE_DIR={BASE}/.uv/cache "
    f"UV_PYTHON_INSTALL_DIR={BASE}/.uv/python UV_INSTALL_DIR={BASE}/.uv/bin UV_NO_MODIFY_PATH=1 "
    f"NUMBA_CACHE_DIR={BASE}/.home/.cache/numba PIP_CACHE_DIR={BASE}/.home/.cache/pip "
    f"PATH={BASE}/.uv/bin:$PATH PYTHONIOENCODING=utf-8"
)


def creds():
    raw = os.environ["MY_SERVER_INFOR"]
    ip, user, password = raw.split("/", 2)
    return ip, user, password


def connect():
    ip, user, password = creds()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username=user, password=password, timeout=20, look_for_keys=False, allow_agent=False)
    return client


def inside(rel: str) -> str:
    path = posixpath.normpath(posixpath.join(BASE, rel))
    if path != BASE and not path.startswith(BASE + "/"):
        raise SystemExit(f"refused: {path} is outside {BASE}")
    return path


def run(cmd: str) -> int:
    client = connect()
    full = f"mkdir -p {BASE} && cd {BASE} && mkdir -p .home .tmp && {ENV} && {cmd}"
    _, stdout, stderr = client.exec_command(full, get_pty=False)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if err:
        sys.stdout.write(err)
    client.close()
    return code


def put(local: str, rel: str) -> None:
    client = connect()
    sftp = client.open_sftp()
    sftp.put(local, inside(rel))
    sftp.close()
    client.close()


def get(rel: str, local: str) -> None:
    client = connect()
    sftp = client.open_sftp()
    sftp.get(inside(rel), local)
    sftp.close()
    client.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    action = sys.argv[1]
    if action == "run":
        sys.exit(run(sys.argv[2]))
    elif action == "put":
        put(sys.argv[2], sys.argv[3])
    elif action == "get":
        get(sys.argv[2], sys.argv[3])
