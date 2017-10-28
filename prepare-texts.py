import logging
import sys
import os
import string

log = logging.getLogger()
log.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)

'''
Simple text files preparation. Merges raw text files into a single one with the format of one document per line.
Converts everything to lower case and removes punctuation. 
'''

dir_in = './aclImdb/train/unsup/'
file_out = './train-unsup.txt'


def main():
    # Translator for stripping punctuation. Not stripping actually but replacing with space to prevent
    # merging words into one. Anyway all multiple spaces will be normalized later.
    strip_table = str.maketrans({key: ' ' for key in string.punctuation})

    log.info("Processing directory %s", dir_in)
    with open(file_out, mode='w', encoding='utf8') as f_out:
        for i, f_name in enumerate(os.listdir(dir_in)):
            if not os.path.isfile(f_name):
                continue
            if f_name.endswith(".txt"):
                f_path = os.path.join(dir_in, f_name)
                if i % 1000 == 0:
                    log.info(str(i) + ":" + f_path)

                # process the text to one string etc
                txt = read_file(f_path)\
                    .replace('<br />', ' ')\
                    .lower()\
                    .translate(strip_table)   # removing punctuation
                f_out.write(' '.join(txt.split()) + '\n')  # removing multiple spaces, tabs and CRLFs

        log.info("*** Written %i bytes", f_out.tell())


def read_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return f.read()


if __name__ == '__main__':
    main()
