import matplotlib.pyplot as plt
import neural_network as nn
import random
import time
import math
import graphics as g


def main():
    neural_network = nn.NeuralNetwork(2, 5, 1)
    start_time = time.time()
    plt.title("Training Data Improvement")
    plt.xlabel("Iterations")
    plt.ylabel("Guess")

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    # number of training iterations
    itr = 10000

    gr = g.Graphics(2, 5, 1)

    for i in range(itr):
        file = open("training_data.txt", "r")
        arr = []

        for j in range(4):
            str = file.readline()
            str_split = str.split(", ")
            arr.append(str_split)
        random.shuffle(arr)

        for j in range(4):
            input = [int(arr[j][0]), int(arr[j][1])]
            target = [int(arr[j][2].strip())]
            neural_network.train(input, target, gr)

        if i % 100 == 1:
            l1.append(neural_network.feed_forward([0, 0]))
            l2.append(neural_network.feed_forward([1, 1]))
            l3.append(neural_network.feed_forward([0, 1]))
            l4.append(neural_network.feed_forward([1, 0]))
        file.close()

        # print progress bar
        print_progress_bar(i + 1, itr, prefix='Progress:', suffix='Complete', length=50)

    # calculate and display training time
    display_time(start_time)

    # testing data for the network
    print("[0, 0]: ", end="")
    print(round(neural_network.feed_forward([0, 0])[0]))
    print("[1, 1]: ", end="")
    print(round(neural_network.feed_forward([1, 1])[0]))
    print("[0, 1]: ", end="")
    print(round(neural_network.feed_forward([0, 1])[0]))
    print("[1, 0]: ", end="")
    print(round(neural_network.feed_forward([1, 0])[0]))

    # plot points and display graph
    plt.plot(l1, "black")
    plt.plot(l2, "black")
    plt.plot(l3, "black")
    plt.plot(l4, "black")
    plt.show()


# prints the progress bar for training data iterations
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New line on Complete
    if iteration == total:
        print()


# displays the time from start_time to the time the function was called
def display_time(start_time):
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("\nTime Elapsed: ", end="")
    if time_elapsed > 60:
        print("{:02d}".format(math.floor(time_elapsed / 60)), end="")
        print(":{:02d}".format(round(time_elapsed % 60)), end="\n\n")
    else:
        print("00:{:02d}".format(round(time_elapsed % 60)), end="\n\n")


if __name__ == "__main__":
    main()