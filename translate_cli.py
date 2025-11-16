#!/usr/bin/env python3

import argparse
import sys
from inference import OdiaTranslator, quick_translate

def main():
    parser = argparse.ArgumentParser(
        description="Odia to English Translation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "ମୁଁ ଭଲ ଅଛି"                    # Translate single text
  %(prog)s --interactive                    # Interactive mode
  %(prog)s --test                          # Run model tests
  %(prog)s --file input.txt                # Translate from file (saves to output.txt)
  %(prog)s --file input.txt -o custom.txt  # Translate from file (saves to custom.txt)
  echo "ଧନ୍ୟବାଦ" | %(prog)s --stdin       # Translate from stdin
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        'text', 
        nargs='?', 
        help='Odia text to translate'
    )
    input_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive translation mode'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Translate text from file (one sentence per line). Creates output.txt automatically if no --output specified.'
    )
    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='Read text from stdin'
    )
    
    # Action options
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run model tests with sample translations'
    )
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='t5_weights_only.pt',
        help='Path to model file (default: t5_weights_only.pt)'
    )
    
    # Output options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - only output translations'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for translations'
    )
    
    args = parser.parse_args()
    
    # Initialize translator
    if not args.quiet:
        print("Initializing Odia-English Translator...")
    
    try:
        translator = OdiaTranslator(args.model)
    except Exception as e:
        print(f"Failed to initialize translator: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle test mode
    if args.test:
        translator.test_model()
        return
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode(translator, args.quiet)
        return
    
    # Handle file input
    if args.file:
        # If no output file specified, create output.txt automatically
        output_file = args.output if args.output else "output.txt"
        translate_file(translator, args.file, output_file, args.quiet)
        return
    
    # Handle stdin input
    if args.stdin:
        translate_stdin(translator, args.output, args.quiet)
        return
    
    # Handle single text translation
    if args.text:
        translation = translator.translate(args.text)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(translation + '\n')
            if not args.quiet:
                print(f"Translation saved to: {args.output}")
        else:
            if args.quiet:
                print(translation)
            else:
                print(f"Odia:       {args.text}")
                print(f"English:    {translation}")
        return
    
    # No input provided
    parser.print_help()

def interactive_mode(translator, quiet=False):
    if not quiet:
        print(f"\n{'='*60}")
        print("INTERACTIVE TRANSLATION MODE")
        print("Type Odia text to translate")
        print("Commands: 'quit', 'exit', 'q' to exit")
        print(f"{'='*60}")
    
    while True:
        try:
            if quiet:
                odia_input = input().strip()
            else:
                odia_input = input("\nOdia: ").strip()
            
            if not odia_input or odia_input.lower() in ['quit', 'exit', 'q']:
                if not quiet:
                    print("Goodbye!")
                break
            
            translation = translator.translate(odia_input)
            
            if quiet:
                print(translation)
            else:
                print(f"English: {translation}")
                
        except KeyboardInterrupt:
            if not quiet:
                print("\nGoodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

def translate_file(translator, input_file, output_file=None, quiet=False):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not quiet:
            print(f"Translating {len(lines)} lines from: {input_file}")
        
        translations = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                if not quiet:
                    print(f"   {i:3d}/{len(lines)}: {line[:50]}{'...' if len(line) > 50 else ''}")
                
                translation = translator.translate(line)
                translations.append(translation)
                
                if not quiet:
                    print(f"        -> {translation}")
            else:
                translations.append("")
        
        # Save or print results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
            if not quiet:
                print(f"Translations saved to: {output_file}")
        else:
            if quiet:
                for translation in translations:
                    print(translation)
            else:
                print(f"\n{'='*60}")
                print("ALL TRANSLATIONS:")
                print(f"{'='*60}")
                for i, (original, translation) in enumerate(zip(lines, translations), 1):
                    if original.strip():
                        print(f"{i:3d}. {original.strip()}")
                        print(f"     -> {translation}")
                        print()
    
    except FileNotFoundError:
        print(f"File not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

def translate_stdin(translator, output_file=None, quiet=False):
    try:
        if not quiet:
            print("Reading from stdin... (Ctrl+D to finish)")
        
        lines = sys.stdin.readlines()
        translations = []
        
        for line in lines:
            line = line.strip()
            if line:
                translation = translator.translate(line)
                translations.append(translation)
            else:
                translations.append("")
        
        # Save or print results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
            if not quiet:
                print(f"Translations saved to: {output_file}")
        else:
            for translation in translations:
                print(translation)
    
    except KeyboardInterrupt:
        if not quiet:
            print("\nInterrupted!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
