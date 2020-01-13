#!/usr/bin/python

import sys
import urllib.request

def main():
    # print command line arguments
    urllib.request.urlretrieve(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()