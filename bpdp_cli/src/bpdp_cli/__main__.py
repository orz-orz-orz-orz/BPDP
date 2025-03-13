import pathlib
import argparse
import torchaudio
import bpdp

def main():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="single input .wav file")
    group.add_argument("-I", "--input_dir", help="input dir contains .wav files")
    
    parser.add_argument("-l", "--wl_0", default=0.05, type=float, help="the window length 0")
    parser.add_argument("-s", "--wl_1", default=0.002, type=float, help="the window length 1")
    parser.add_argument("-L", "--f_lo", default=50.0, type=float, help="the minimum f0")
    parser.add_argument("-S", "--f_hi", default=550.0, type=float, help="the maximum f0")
    parser.add_argument("-b", "--beam_size", default=5, type=int, help="the beam size")
    parser.add_argument("-f", "--filter", default="bp1", type=str, help="the filtering method")
    parser.add_argument("-r", "--resampler", default="makima", type=str, help="the resampling method")

    parser.add_argument("--suffix", default=".bpdp.txt", type=str, help="the suffix of output files")
    parser.add_argument("--verbose", action='store_true', help="print every step")
    parser.add_argument("--dry_run", action='store_true', help="dry run, no file will be created")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Args: {args}")

    if args.input is not None:
        in_file = pathlib.Path(args.input)
        out_file = in_file.with_suffix(args.suffix)
        
        if args.verbose:
            print(f"Extract {in_file} -> {out_file}")

        if not in_file.exists():
            raise FileNotFoundError(f"{in_file} was not found")
        
        if not args.dry_run:
            wav, sr = torchaudio.load(str(in_file))
            epochs = bpdp.bpdp(wav[0], sr, args.wl_0, args.wl_1, args.f_lo, args.f_hi, args.beam_size, args.filter, args.resampler)
            with out_file.open("w") as f:
                for epoch in epochs:
                    f.write(f"{epoch}\n")

    elif args.input_dir is not None:
        in_dir = pathlib.Path(args.input_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"{in_dir} was not found")

        for in_file in in_dir.glob("**/*.wav"):
            out_file = in_file.with_suffix(args.suffix)
            if args.verbose:
                print(f"Extract {in_file} -> {out_file}")
            if not args.dry_run:
                wav, sr = torchaudio.load(str(in_file))
                epochs = bpdp.bpdp(wav[0], sr, args.wl_0, args.wl_1, args.f_lo, args.f_hi, args.beam_size, args.filter, args.resampler)
                with out_file.open("w") as f:
                    for epoch in epochs:
                        f.write(f"{epoch}\n")

if __name__ == "__main__":
    main()

