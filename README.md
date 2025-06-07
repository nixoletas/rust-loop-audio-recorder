# Rust Audio Recorder

![image](https://github.com/user-attachments/assets/ff277848-f0a5-420b-9322-569c2f9d5b6f)


A simple audio recorder application built with Rust, featuring a graphical user interface.

## Features

- Record audio from your default input device
- Save recordings as WAV files
- Simple and intuitive GUI
- Cross-platform support

## Requirements

- Rust and Cargo installed on your system
- A working audio input device (microphone)

## Building and Running

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rust-audio-recorder.git
cd rust-audio-recorder
```

2. Build and run the application:
```bash
cargo run --release
```

## Usage

1. Click "Start Recording" to begin recording audio
2. Click "Stop Recording" to stop the recording
3. Click "Save Recording" to save the recording as a WAV file (saved as "recording.wav" in the current directory)

## Dependencies

- eframe: GUI framework
- cpal: Cross-platform audio I/O
- ringbuf: Ring buffer for audio data
- hound: WAV file handling
- anyhow: Error handling

## License

MIT License 
