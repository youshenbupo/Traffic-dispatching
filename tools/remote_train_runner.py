"""Keep an SSH channel open for one seeded remote training run."""
import os
import sys

import paramiko


def main():
    config, gpu, seed, exp_name, log_path = sys.argv[1:6]
    pretrained = sys.argv[6] if len(sys.argv) > 6 else None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "172.16.90.2",
        username="jyh",
        password=os.environ["RL_SSH_PASSWORD"],
    )
    command = (
        "cd /data/users/jyh/light_RL && "
        "source /opt/miniforge3/etc/profile.d/conda.sh && "
        "conda activate light_rl && "
        f"python scripts/train.py --config {config} --gpus {gpu} "
        f"--seed {seed} --exp_name {exp_name} "
        + (f"--pretrained_model {pretrained} " if pretrained else "")
        + f"> {log_path} 2>&1"
    )
    _, stdout, _ = client.exec_command(f"bash -lc {command!r}")
    status = stdout.channel.recv_exit_status()
    client.close()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
