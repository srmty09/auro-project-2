

import argparse
import sys
from inference import OdiaTranslator, quick_translate

def main():
    parser = argparse.ArgumentParser(
        description="Odia to English Translation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        epilog="""
Examples:
  %(prog)s "ମୁଁ ଭଲ ଅଛି"
  %(prog)s --interactive
  %(prog)s --test
  %(prog)s --file input.txt
  %(prog)s --file input.txt -o custom.txt
  echo "ଧନ୍ୟବାଦ" | %(prog)s --stdin