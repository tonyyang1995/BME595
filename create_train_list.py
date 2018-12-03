import os
import sys
import argparse

def create_train_list(root, name):
    trainlist = open(name, 'w')
    data_root = os.listdir(root)
    for f in data_root:
        imgnames = os.listdir(os.path.join(root,f))
        for path in imgnames:
            imgs = os.listdir(os.path.join(root, f, path))
            for img in imgs:
                trainlist.write(os.path.join(f,path,img) + '\n')

def create_val_list(root, name):
    data_root = os.listdir(root)
    if not os.path.exists('Lists/'):
        os.makedirs('Lists/')
    vallist = open('Lists/'+ name +'.txt', 'w')

    for f in data_root:
        imgnames = os.listdir(os.path.join(root, f))
        for img in imgnames:
            vallist.write(os.path.join(f, img) + '\n')

    vallist.close()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', metavar='root', help='root of the dataset')
    parser.add_argument('--name', metavar='name', default='trainlist.txt', help='name of the trainlist, testlist or vallist')
    parser.add_argument('--isval', metavar='isval', default=1)
    args = parser.parse_args()
    print(args.isval)
    if args.isval:
        create_val_list(args.root, args.name)
    else:
        create_train_list(args.root, args.name)
