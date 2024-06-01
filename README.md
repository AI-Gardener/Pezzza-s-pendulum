# Pezzza-s-pendulum
Following https://www.youtube.com/watch?v=EvV5Qtp_fYg

If you want ot export to a video, create a folder named 'output' beforehand.
## Useful commands
```bash
cargo run
cargo run --release
# Concatenating frames
ffmpeg -r 60 -pattern_type glob -i 'output/*.png' -c:v libx264 -crf 10 -preset veryslow -y out.mp4
```