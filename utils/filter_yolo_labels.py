import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='path', help='Set detector to use', default='SIFT_CUSTOM', type=str.upper)
args = parser.parse_args()

labels_to_remove = [0]
lables_to_convert = [(1, ), (2, ), (1, ), (1, ), (1, )]


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        filelines = f.readlines()
        f.seek(0)
        for line in filelines:
            for rm_label in labels_to_remove:
                if line.startswith(rm_label):
                    continue

                for old_label, new_label in lables_to_convert:
                    if line.startswith(old_label):
                        new_label = new_label + line[len(old_label):]
                        f.write(new_label + "\n")


def main():
    path = ''
    os.chdir(path)

    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"

            read_text_file(file_path)