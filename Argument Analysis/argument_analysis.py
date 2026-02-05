import sys

if __name__ == "__main__":
    print("\nNumber of arguments = ", len(sys.argv))

for i in range(len(sys.argv)):
    print("Arg[", i, "] = ", sys.argv[i])