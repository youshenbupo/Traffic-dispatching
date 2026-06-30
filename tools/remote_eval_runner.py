"""Keep one SSH connection open for a sequence of remote evaluations."""
import os
import sys

import paramiko


def main():
    model_path, gpu, seed_start = sys.argv[1:4]
    pairs = [item.split("=", 1) for item in sys.argv[4:]]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "172.16.90.2",
        username="jyh",
        password=os.environ["RL_SSH_PASSWORD"],
    )
    for config, output in pairs:
        command = (
            "cd /data/users/jyh/light_RL && "
            "source /opt/miniforge3/etc/profile.d/conda.sh && "
            "conda activate light_rl && "
            f"python scripts/eval.py --config {config} "
            f"--model_path {model_path} --gpus {gpu} "
            f"--num_episodes 5 --seed_start {seed_start} --output {output}"
        )
        _, stdout, stderr = client.exec_command(f"bash -lc {command!r}")
        status = stdout.channel.recv_exit_status()
        if status:
            sys.stderr.write(stderr.read().decode("utf-8", errors="replace"))
            client.close()
            raise SystemExit(status)
    client.close()


if __name__ == "__main__":
    main()
