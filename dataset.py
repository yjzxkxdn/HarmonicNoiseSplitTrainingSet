import torch    
import random
import numpy as np


class GeneratedHarmonicNoiseSplitTrainingSet(torch.utils.data.Dataset):
    def __init__(self, 
            harmonic_generator,
            noise_generator,
            sr, 
            hop_length, 
            cropsize, 
            copy_rate,
            mixup_rate, 
            mixup_alpha,
            mode='noise_as_inst'  # 'noise_as_inst'或'harmonic_as_inst'
        ):
        self.harmonic_generator = harmonic_generator
        self.noise_generator = noise_generator
        self.sr = sr
        self.hop_length = hop_length
        self.cropsize = cropsize
        self.copy_rate = copy_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.mode = mode
        
        # 计算波形时长（秒）
        self.waveform_sec = (cropsize - 1) * hop_length / sr
        
    def __len__(self):
        return 1000

    def generate_pair(self):
        """生成一对谐波和噪声信号"""
        harmonic = self.harmonic_generator.generate_harmonic(
            n_samples=int(self.waveform_sec * self.sr)
        )
        
        noise = self.noise_generator.generate_noise(
            n_samples=int(self.waveform_sec * self.sr)
        )
        
        min_len = min(len(harmonic), len(noise))
        harmonic = harmonic[:min_len]
        noise = noise[:min_len]
        
        harmonic = np.asarray([harmonic], dtype=np.float32)
        noise = np.asarray([noise], dtype=np.float32)
        
        return harmonic, noise

    def do_aug(self, X, y):
        max_amp = np.max([np.abs(X).max(), np.abs(y).max()])
        max_shift = min(1, np.log10(1 / max_amp))
        log10_shift = random.uniform(-1, max_shift)
        X =  X * (10 ** log10_shift)
        y =  y * (10 ** log10_shift)

        if np.random.uniform() < self.copy_rate:
            # inst
            X = y.copy()

        return X, y

    def do_mixup(self, X, y):
        idx = np.random.randint(0, len(self))
        X_i, y_i = self.__getitem__(idx, skip_mix=True)
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        X = lam * X + (1 - lam) * X_i
        y = lam * y + (1 - lam) * y_i

        return X, y

    def __getitem__(self, idx, skip_mix=False):
        # 生成一对信号
        harmonic, noise = self.generate_pair()
        
        # 根据模式确定混合物和乐器
        if self.mode == 'noise_as_inst':
            # 噪声作为乐器，噪声+谐波作为混合物
            y = noise
            X = noise + harmonic
        elif self.mode == 'harmonic_as_inst':  # 'harmonic_as_inst'
            # 谐波作为乐器，噪声+谐波作为混合物
            y = harmonic
            X = harmonic + noise
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))
        
        # 数据增强
        X, y = self.do_aug(X, y)
        
        # Mixup增强
        if not skip_mix and np.random.uniform() < self.mixup_rate:
            X, y = self.do_mixup(X, y)
            
        return X, y


class GeneratedHarmonicNoiseSplitValidationSet(torch.utils.data.Dataset):
    def __init__(self, 
            harmonic_generator,
            noise_generator,
            sr, 
            cropsize,
            val_size=10,  # 验证集大小
            mode='noise_as_inst'  # 'noise_as_inst'或'harmonic_as_inst'
        ):
        self.harmonic_generator = harmonic_generator
        self.noise_generator = noise_generator
        self.sr = sr
        self.val_size = val_size
        self.mode = mode
        
        # 计算波形时长（秒）
        self.waveform_sec = (cropsize - 1) * cropsize / sr
        
        # 预生成验证集
        self.cache = []
        for _ in range(val_size):
            harmonic, noise = self.generate_pair()
            
            if self.mode == 'noise_as_inst':
                y = noise
                X = noise + harmonic
            elif self.mode == 'harmonic_as_inst':  # 'harmonic_as_inst'
                y = harmonic
                X = harmonic + noise
            else:
                raise ValueError("Invalid mode: {}".format(self.mode))
                
            
            self.cache.append((X, y))

    def generate_pair(self):
        """生成一对谐波和噪声信号"""
        harmonic = self.harmonic_generator.generate_harmonic(
            n_samples=int(self.waveform_sec * self.sr)
        )
        noise = self.noise_generator.generate_noise(
            n_samples=int(self.waveform_sec * self.sr)
        )
        
        min_len = min(len(harmonic), len(noise))
        harmonic = harmonic[:min_len]
        noise = noise[:min_len]
        
        harmonic = np.asarray([harmonic], dtype=np.float32)
        noise = np.asarray([noise], dtype=np.float32)
        

        return harmonic, noise
    def __len__(self):
        return self.val_size

    def __getitem__(self, idx):
        return self.cache[idx]