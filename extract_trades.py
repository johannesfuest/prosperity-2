import argparse


def main(filename):
    # open the file and find the line with the trades
    # Trade History:
    with open(filename, 'r') as f:
        csv_start = False
        csv_end = False
        for line in f:
            if line.startswith('Activities log:'):
                csv_start = True
                break
        with open('data/rounds.csv', 'w') as f_out_1:
            for line in f:
                if csv_start and not csv_end:
                    f_out_1.write(line)
                if line.strip() == "":
                    csv_end = True 
        
        for line in f:
            if line.startswith('Trade History:'):
                break

        # store the rest of the file starting from the next line in a string and dump it to a json file
        with open('data/trades.json', 'w') as f_out:
            for line in f:
                f_out.write(line)

if __name__ == "__main__":
    # parse the filename
    parser = argparse.ArgumentParser(description='Extract trades from a file')
    parser.add_argument('filename', help='the file to extract trades from')
    args = parser.parse_args()
    main(args.filename)
