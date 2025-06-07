use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::collections::VecDeque;
use egui_plot::{Plot, Line, PlotPoints, VLine};
use eframe::emath::Vec2b;

// Constants for loop recording
const BPM: f32 = 120.0;
const BARS: f32 = 4.0;
const BEATS_PER_BAR: f32 = 4.0;
const SECONDS_PER_MINUTE: f32 = 60.0;

// Colors for different loops
const WAVEFORM_COLORS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(100, 200, 100),  // Green
    egui::Color32::from_rgb(200, 100, 100),  // Red
    egui::Color32::from_rgb(100, 100, 200),  // Blue
    egui::Color32::from_rgb(200, 200, 100),  // Yellow
    egui::Color32::from_rgb(200, 100, 200),  // Purple
    egui::Color32::from_rgb(100, 200, 200),  // Cyan
    egui::Color32::from_rgb(200, 150, 100),  // Orange
    egui::Color32::from_rgb(150, 200, 150),  // Light Green
];

#[derive(Clone)]
struct Recording {
    samples: Vec<f32>,
    sample_rate: u32,
    color_index: usize,
}

struct AudioRecorder {
    is_recording: Arc<AtomicBool>,
    recordings: Arc<Mutex<Vec<Recording>>>,
    current_recording: Arc<Mutex<Recording>>,
    stream: Option<cpal::Stream>,
    error_message: Option<String>,
    visualization_buffers: Arc<Mutex<Vec<VecDeque<f32>>>>,
    volume_amplification: f32,
    samples_recorded: Arc<Mutex<usize>>,
    current_color_index: Arc<Mutex<usize>>,
}

impl Default for AudioRecorder {
    fn default() -> Self {
        let sample_rate = 44100; // Default sample rate
        let loop_duration_samples = ((BARS * BEATS_PER_BAR * SECONDS_PER_MINUTE / BPM) * sample_rate as f32) as usize;

        Self {
            is_recording: Arc::new(AtomicBool::new(false)),
            recordings: Arc::new(Mutex::new(Vec::new())),
            current_recording: Arc::new(Mutex::new(Recording {
                samples: Vec::with_capacity(loop_duration_samples),
                sample_rate,
                color_index: 0,
            })),
            stream: None,
            error_message: None,
            visualization_buffers: Arc::new(Mutex::new(vec![VecDeque::with_capacity(100)])),
            volume_amplification: 10.0,
            samples_recorded: Arc::new(Mutex::new(0)),
            current_color_index: Arc::new(Mutex::new(0)),
        }
    }
}

impl AudioRecorder {
    fn start_recording(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let _channels = config.channels();

        // Calculate loop duration in samples
        let loop_duration_samples = ((BARS * BEATS_PER_BAR * SECONDS_PER_MINUTE / BPM) * sample_rate as f32) as usize;

        let is_recording = self.is_recording.clone();
        let recordings = self.recordings.clone();
        let current_recording = self.current_recording.clone();
        let visualization_buffers = self.visualization_buffers.clone();
        let volume_amplification = self.volume_amplification;
        let samples_recorded = self.samples_recorded.clone();
        let current_color_index = self.current_color_index.clone();

        let err_fn = move |err| {
            eprintln!("an error occurred on stream: {}", err);
        };

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if is_recording.load(Ordering::Relaxed) {
                    let mut current = current_recording.lock().unwrap();
                    let mut recorded = samples_recorded.lock().unwrap();

                    // Apply volume amplification to the audio data
                    let amplified_data: Vec<f32> = data.iter()
                        .map(|&x| (x * volume_amplification).clamp(-1.0, 1.0))
                        .collect();

                    // Add new samples to current recording
                    current.samples.extend_from_slice(&amplified_data);
                    *recorded += amplified_data.len();

                    // Check if we've reached the loop duration
                    if *recorded >= loop_duration_samples {
                        // Save current recording and start a new one
                        let mut recordings = recordings.lock().unwrap();
                        let current_samples = current.samples.clone();
                        let color_idx = *current_color_index.lock().unwrap();
                        recordings.push(Recording {
                            samples: current_samples,
                            sample_rate,
                            color_index: color_idx,
                        });

                        // Reset current recording
                        *recorded = 0;
                        current.samples.clear();
                        
                        // Update color index for next loop
                        let mut color_idx = current_color_index.lock().unwrap();
                        *color_idx = (*color_idx + 1) % WAVEFORM_COLORS.len();
                        
                        // Add new visualization buffer for next loop
                        let mut buffers = visualization_buffers.lock().unwrap();
                        buffers.push(VecDeque::with_capacity(100));
                    }

                    let rms = (amplified_data.iter().map(|&x| x * x).sum::<f32>() / amplified_data.len() as f32).sqrt();
                    let mut buffers = visualization_buffers.lock().unwrap();
                    if let Some(current_buffer) = buffers.last_mut() {
                        current_buffer.push_back(rms);
                        if current_buffer.len() > 100 {
                            current_buffer.pop_front();
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);
        self.is_recording.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop_recording(&mut self) {
        self.is_recording.store(false, Ordering::Relaxed);
        if let Some(stream) = self.stream.take() {
            let _ = stream.pause();
        }
    }

    fn save_recording(&self, filename: &str) -> Result<()> {
        let recordings = self.recordings.lock().unwrap();
        if recordings.is_empty() {
            return Err(anyhow::anyhow!("No audio data to save"));
        }

        // Mix all recordings together
        let mut mixed_samples = Vec::new();
        for recording in recordings.iter() {
            if mixed_samples.is_empty() {
                mixed_samples = recording.samples.clone();
            } else {
                // Mix the recordings by adding samples and normalizing
                for (i, &sample) in recording.samples.iter().enumerate() {
                    if i < mixed_samples.len() {
                        mixed_samples[i] = (mixed_samples[i] + sample).clamp(-1.0, 1.0);
                    }
                }
            }
        }

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: recordings[0].sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(filename, spec)?;
        for &sample in mixed_samples.iter() {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

// Add a helper function for downsampling
fn downsample(samples: &[f32], target_points: usize) -> Vec<f32> {
    if samples.len() <= target_points {
        return samples.to_vec();
    }
    let chunk_size = samples.len() as f32 / target_points as f32;
    (0..target_points)
        .map(|i| {
            let start = (i as f32 * chunk_size).floor() as usize;
            let end = ((i as f32 + 1.0) * chunk_size).ceil() as usize;
            let end = end.min(samples.len());
            if start < end {
                samples[start..end].iter().copied().sum::<f32>() / (end - start) as f32
            } else {
                samples[start]
            }
        })
        .collect()
}

impl eframe::App for AudioRecorder {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repainting while recording
        if self.is_recording.load(Ordering::Relaxed) {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);  // Add some padding at the top
                ui.heading(egui::RichText::new("Audio Recorder").size(32.0));
            
            if let Some(error) = &self.error_message {
                ui.colored_label(egui::Color32::RED, error);
            }

                ui.add_space(20.0);  // Add spacing between elements

                // Volume control in a centered container
                ui.horizontal(|ui| {
                    ui.add_space(ui.available_width() / 4.0);  // Center the volume control
                    ui.label(egui::RichText::new("Volume Amplification:").size(18.0));
                    let slider = egui::Slider::new(&mut self.volume_amplification, 1.0..=10.0);
                    ui.add_space(10.0);
                    if ui.add(slider).changed() {
                        // The value will be used in the next audio callback
                    }
                    ui.add_space(10.0);
                });

                ui.add_space(20.0);  // Add spacing between elements

                // Center the buttons
                ui.horizontal(|ui| {
                    ui.add_space(ui.available_width() / 3.0);  // Center the buttons
            if !self.is_recording.load(Ordering::Relaxed) {
                        if ui.add(egui::Button::new(egui::RichText::new("Start Recording").size(20.0))
                            .min_size(egui::vec2(200.0, 50.0))).clicked() {
                    if let Err(e) = self.start_recording() {
                        self.error_message = Some(format!("Failed to start recording: {}", e));
                    }
                }
            } else {
                        if ui.add(egui::Button::new(egui::RichText::new("Stop Recording").size(20.0))
                            .min_size(egui::vec2(200.0, 50.0))).clicked() {
                    self.stop_recording();
                }
            }
                });

                ui.add_space(20.0);  // Add spacing before waveform

                // Center the waveform plot
                ui.horizontal(|ui| {
                    ui.add_space(ui.available_width() / 4.0);  // Center the plot
                    // Copy data out of mutexes to avoid holding locks during plotting
                    let recordings = {
                        let recs = self.recordings.lock().unwrap();
                        recs.clone()
                    };
                    let current_samples = {
                        let rec = self.current_recording.lock().unwrap();
                        rec.samples.clone()
                    };
                    let color_index = *self.current_color_index.lock().unwrap();
                    let samples_recorded = *self.samples_recorded.lock().unwrap();
                    let loop_duration_samples = ((BARS * BEATS_PER_BAR * SECONDS_PER_MINUTE / BPM) * 44100.0) as usize;
                    let plot_points = 600;
                    let plot_width = plot_points as f64;

                    Plot::new("audio_waveform")
                        .view_aspect(3.0)
                        .height(160.0)
                        .auto_bounds(Vec2b::TRUE)
                        .show_x(false)
                        .show_y(false)
                        .show(ui, |plot_ui| {
                            // Draw grid lines for bars
                            let bars = BARS as usize;
                            for bar in 1..bars {
                                let x = (bar as f64) * plot_width / BARS as f64;
                                plot_ui.vline(VLine::new(x).color(egui::Color32::DARK_GRAY));
                            }
                            // Draw loop end
                            plot_ui.vline(VLine::new(plot_width).color(egui::Color32::LIGHT_GRAY).style(egui_plot::LineStyle::Dashed { length: 4.0 }));

                            // Draw each recording's waveform with gradient and transparency
                            for (i, rec) in recordings.iter().enumerate() {
                                let samples = &rec.samples;
                                if samples.is_empty() { continue; }
                                let down = downsample(samples, plot_points);
                                let color1 = WAVEFORM_COLORS[rec.color_index % WAVEFORM_COLORS.len()];
                                let alpha = 64 + (128 / (i as u8 + 1));
                                let points: Vec<[f64; 2]> = down.iter().enumerate().map(|(j, &v)| {
                                    let x = (j as f64) * plot_width / (plot_points as f64 - 1.0);
                                    [x, (v * 500.0) as f64]
                                }).collect();
                                let line = Line::new(PlotPoints::from(points)).color(color1.linear_multiply(alpha as f32 / 255.0));
                                plot_ui.line(line);
                            }
                            // Draw current recording waveform (active layer)
                            if !current_samples.is_empty() {
                                let down = downsample(&current_samples, plot_points);
                                let color = WAVEFORM_COLORS[color_index % WAVEFORM_COLORS.len()];
                                let points: Vec<[f64; 2]> = down.iter().enumerate().map(|(j, &v)| {
                                    let x = (j as f64) * plot_width / (plot_points as f64 - 1.0);
                                    [x, (v * 500.0) as f64]
                                }).collect();
                                let line = Line::new(PlotPoints::from(points)).color(color.linear_multiply(0.9));
                                plot_ui.line(line);
                            }
                            // Draw glowing playhead
                            if self.is_recording.load(Ordering::Relaxed) {
                                let playhead_x = (samples_recorded as f64) * plot_width / (loop_duration_samples as f64);
                                plot_ui.vline(VLine::new(playhead_x)
                                    .color(egui::Color32::from_rgb(0, 255, 180))
                                    .width(2.5));
                            }
                        });
                });

                ui.add_space(20.0);  // Add spacing before save button

                // Center the save button
                ui.horizontal(|ui| {
                    ui.add_space(ui.available_width() / 3.0);  // Center the save button
                    if !self.is_recording.load(Ordering::Relaxed) {
                        if !self.recordings.lock().unwrap().is_empty() {
                            if ui.add(egui::Button::new(egui::RichText::new("Save Recording").size(20.0))
                                .min_size(egui::vec2(200.0, 50.0))).clicked() {
                    if let Err(e) = self.save_recording("recording.wav") {
                        self.error_message = Some(format!("Failed to save recording: {}", e));
                    } else {
                        self.error_message = Some("Recording saved successfully!".to_string());
                    }
                }
            }
                    }
                });
            });
        });
    }
}

fn main() {
    let options = eframe::NativeOptions::default();
    
    if let Err(e) = eframe::run_native(
        "Audio Recorder",
        options,
        Box::new(|_cc| Box::new(AudioRecorder::default())),
    ) {
        eprintln!("Error running application: {}", e);
    }
} 