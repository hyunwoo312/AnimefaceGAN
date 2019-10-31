from time import sleep
import sys
from GAN import GenerativeAdversarialNetwork
####################
def main():
    print(
'''
Running AnimefaceGAN. . .
Not the highest quality nor the most creative, but y'know. . .
by Hyun
'''
    )
    sleep(3.0)
    print(
'''
If you would like to upload your own test samples and redo the training,
change directory to GAN/datafiles/waifus then manage the files to your
own will. 

If you would like to see the training in action/just redo the training
from the beginning, type 'train' then press enter.

If you would like to see sample images from the pre-trained model,
type 'waifus' then press enter.

If you would like to quit this program, type 'quit' then press enter.
'''
    )
    print_cool = "☆*:.｡.o(≧▽≦)o.｡.:*☆"
    for _ in range(len(print_cool)):
        sys.stdout.write("\r{0}".format(" "*_ + print_cool[_]))
        sys.stdout.flush()
        sleep(0.15)
        if _ == len(print_cool) - 1:
            sys.stdout.write("\033[2K\033[1G")
            sys.stdout.write("\r{0}\n".format("☆*:.｡.o(≧▽≦)o.｡.:*☆"))
    print(">>>", end='')
    option = input()
    if option == "train":
        with GenerativeAdversarialNetwork as log:
            pass
    elif option == "waifus":
        with GenerativeAdversarialNetwork as gen:
            pass
    elif option == 'q' or option == "quit":
        return # exits the program
    else:
        print("Invalid option. . . restarting")
        sleep(1)
        main()
    

if __name__ == "__main__":
    main()
    