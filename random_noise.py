import numpy as np
import librosa
from scipy.interpolate import griddata
import soundfile as sf
from tools import sample_y_coords_with_weighted_intervals

class GeneratorNoise:
    def __init__(self, sr=44100, hopsize=512, random_seed=527):
        self.sr = sr
        self.hopsize = hopsize
        self.n_fft = 2048  # 固定FFT大小为2048
        self.rng = np.random.default_rng(random_seed)
        
    def generate_noise(self, n_samples):
        noise = self.rng.standard_normal(n_samples)
        
        noise_stft = librosa.stft(
            noise, 
            n_fft=self.n_fft, 
            hop_length=self.hopsize, 
            win_length=self.n_fft
        )
        
        spec_len = noise_stft.shape[1]
        spec_high = noise_stft.shape[0]
        
        # 生成振幅谱
        ampl_spec, x_coords, y_coords = self.generate_amplitudes(spec_len, spec_high)
        
        # 应用振幅谱到噪声STFT
        noise_stft = ampl_spec.T * noise_stft
        
        noise = librosa.istft(
            noise_stft, 
            hop_length=self.hopsize, 
            win_length=self.n_fft
        )
        
        return noise
        
    def generate_amplitudes(self, spec_len, spec_high):
        ampl_spec = np.zeros((spec_len, spec_high))
        
        # 控制点密度：每秒钟的控制点数量
        ampl_control_points_density = 16  
        seconds = spec_len * self.hopsize / self.sr
        num_random_points = int(ampl_control_points_density * seconds)
        
        x_min, x_max = 0, spec_len
        y_min, y_max = 0, spec_high
        
        x_coords_rand = self.rng.choice(
            np.arange(x_min + 1, x_max - 1), 
            size=num_random_points, 
            replace=False
        )
        
        y_coords_rand = sample_y_coords_with_weighted_intervals(
            self.rng, y_min, y_max, num_random_points
        )
        
        # 添加固定点（边界和中心点）
        fixed_points = np.array([
            [x_min, y_min],  # 左下
            [x_max - 1, y_min],  # 右下
            [x_min, y_max - 1],  # 左上
            [x_max - 1, y_max - 1],  # 右上
            [x_max // 2, y_max - 1],  # 上边中点
            [x_max // 2, y_min],  # 下边中点
            [x_min, y_max // 2],  # 左边中点
            [x_max - 1, y_max // 2],  # 右边中点
        ])
        
        # 合并随机点和固定点
        x_coords = np.concatenate([x_coords_rand, fixed_points[:, 0]])
        y_coords = np.concatenate([y_coords_rand, fixed_points[:, 1]])
        
        # 创建控制点坐标
        points = np.column_stack((x_coords, y_coords))
        
        # 为控制点生成随机振幅值（dB范围）
        num_control_points = len(x_coords)
        ampl_values = self.rng.uniform(-50, 0, size=num_control_points)
        
        grid_x, grid_y = np.mgrid[0:spec_len, 0:spec_high]
        
        ampl_spec = griddata(
            points, 
            ampl_values, 
            (grid_x, grid_y), 
            method='linear'
        )
        ampl_spec = np.nan_to_num(ampl_spec)
        
        # 添加低频滚降效果
        low_pass = np.linspace(0, self.rng.uniform(-50, -25), spec_high)
        ampl_spec += low_pass[None, :]
        
        # 转换为线性振幅（从dB）
        ampl_spec = 10 ** (ampl_spec / 20)
        
        return ampl_spec, x_coords, y_coords
    
    
if __name__ == '__main__':
    gen = GeneratorNoise()
    for i in range(10):
        waveform = gen.generate_noise(n_samples=44100 * 5)
        print(waveform.shape)
        sf.write(f'output_noise{i}.wav', waveform, 44100, subtype='FLOAT')