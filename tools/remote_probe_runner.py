"""Keep an SSH channel open for one remote policy-sensitivity probe."""
import os
import sys

import paramiko


def main():
    config, model_path, gpu, output = sys.argv[1:5]
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
        f"python scripts/probe_policy_sensitivity.py --config {config} "
        f"--model_path {model_path} --gpus {gpu} "
        f"--num_episodes 5 --seed_start 35000 --output {output}"
    )
    _, stdout, stderr = client.exec_command(f"bash -lc {command!r}")
    status = stdout.channel.recv_exit_status()
    if status:
        sys.stderr.write(stderr.read().decode("utf-8", errors="replace"))
    client.close()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
