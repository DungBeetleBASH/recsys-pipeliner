import subprocess
from sklearn_container.mms import start_server


def main():
    start_server()

    # prevent docker exit
    subprocess.call(["tail", "-f", "/dev/null"])

if __name__ == "__main__":
    main()