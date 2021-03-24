import sys

avg = 990


def make_balance(src, dst):
    class_list = list([] for x in range(400))  # len=400

    # get orgin list sort by class
    with open(src) as f:
        src_list = [x.strip() for x in f.readlines()]
    for vid in src_list:
        label = int(vid.split(' ')[-1])
        class_list[label].append(vid)

    # get balance list
    for i in range(400):
        while (avg - len(class_list[i])):
            sup = avg - len(class_list[i])
            class_list[i].extend(class_list[i][:sup])

    # output balance list
    with open(dst, 'a+') as f:
        for i in range(400):
            for j in range(len(class_list[i])):
                vid_info = class_list[i][j]
                f.write(vid_info + '\n')


if __name__ == "__main__":
    list_name = sys.argv[1]
    balance_name = list_name.split('.')[0] + '_balance.txt'
    make_balance(list_name, balance_name)
