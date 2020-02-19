import os


def main():
    working = True

    while working:
        print_header()
        cmd = get_user_input()
        working = interpret_command(cmd)


def print_header():
    print('---------------------------------------------------')
    print('                    COHRINT')
    print('               Hide and Seek DDF ')
    print('                  Jack Center ')
    print('---------------------------------------------------')
    print()


def get_user_input():
    print('This program runs the following exercises:')
    print(' [1]: Stationary')
    print()

    cmd = input(' Select an exercise would you like to run: ')
    cmd = cmd.strip().lower()

    return cmd


def interpret_command(cmd):
    if cmd == '1':      # path planning
        os.system("python stationary_program.py")

    else:
        print(' ERROR: unexpected command...')

    print()
    run_again = input(' Would you like to run another exercise?[y/n]: ')

    if run_again != 'y':
        print(" closing program ... goodbye!")
        return False

    return True


if __name__ == '__main__':
    main()
