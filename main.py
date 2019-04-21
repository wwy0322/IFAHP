from framework import Framework
from config import conf_file

def main():
    f = Framework(conf_file)
    f.build()
    if not f.after_build():
        raise RuntimeError("Framework build Fail!")

    print(f)

if __name__ == '__main__':
    main()