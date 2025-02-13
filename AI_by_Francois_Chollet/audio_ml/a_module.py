import librosa
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class AudioDatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = []
        self.stats_df = None
        
    def load_jsonl(self, jsonl_path: str):
        """Загрузка JSONL файла с метаданными"""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """Анализ характеристик аудиофайла"""
        y, sr = librosa.load(audio_path, sr=None)
        
        # Основные характеристики
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0].mean()
        zero_crossings = librosa.zero_crossings(y).sum()
        
        # Спектральные характеристики
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'rms_energy': float(rms),
            'zero_crossing_rate': float(zero_crossings) / len(y),
            'spectral_centroid': float(spec_centroid),
            'spectral_rolloff': float(spec_rolloff),
            'silence_percentage': self._calculate_silence_percentage(y, sr)
        }
    
    def _calculate_silence_percentage(self, y: np.ndarray, sr: int, 
                                   threshold_db: float = -40) -> float:
        """Расчёт процента тишины в аудио"""
        threshold_amplitude = librosa.db_to_amplitude(threshold_db)
        silent_frames = np.sum(np.abs(y) < threshold_amplitude)
        return (silent_frames / len(y)) * 100

    def filter_dataset(self, 
                      min_duration: Optional[float] = None,
                      max_duration: Optional[float] = None,
                      min_rms: Optional[float] = None,
                      max_silence_percentage: Optional[float] = None,
                      min_sample_rate: Optional[int] = None) -> List[Dict]:
        """Фильтрация датасета по заданным критериям"""
        filtered_data = []
        
        for item in self.data:
            audio_path = self.dataset_path / item['audio_filepath']
            try:
                audio_stats = self.analyze_audio(str(audio_path))
                
                # Применяем фильтры
                if min_duration and audio_stats['duration'] < min_duration:
                    continue
                if max_duration and audio_stats['duration'] > max_duration:
                    continue
                if min_rms and audio_stats['rms_energy'] < min_rms:
                    continue
                if max_silence_percentage and audio_stats['silence_percentage'] > max_silence_percentage:
                    continue
                if min_sample_rate and audio_stats['sample_rate'] < min_sample_rate:
                    continue
                
                # Добавляем статистики к метаданным
                item.update(audio_stats)
                filtered_data.append(item)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue
        
        return filtered_data

    def analyze_dataset(self):
        """Анализ всего датасета и подготовка данных для визуализации"""
        all_stats = []
        
        for item in self.data:
            try:
                audio_path = self.dataset_path / item['audio_filepath']
                stats = self.analyze_audio(str(audio_path))
                all_stats.append(stats)
            except Exception as e:
                print(f"Error analyzing {audio_path}: {str(e)}")
                continue
        
        self.stats_df = pd.DataFrame(all_stats)
        return self.get_dataset_statistics()

    def get_dataset_statistics(self) -> Dict:
        """Получение статистики по всему датасету"""
        if self.stats_df is None:
            self.analyze_dataset()
            
        return {
            'total_files': len(self.stats_df),
            'duration': {
                'mean': self.stats_df['duration'].mean(),
                'std': self.stats_df['duration'].std(),
                'min': self.stats_df['duration'].min(),
                'max': self.stats_df['duration'].max()
            },
            'rms_energy': {
                'mean': self.stats_df['rms_energy'].mean(),
                'std': self.stats_df['rms_energy'].std()
            },
            'silence_percentage': {
                'mean': self.stats_df['silence_percentage'].mean(),
                'std': self.stats_df['silence_percentage'].std()
            },
            'sample_rates': self.stats_df['sample_rate'].value_counts().to_dict()
        }

    def plot_distributions(self, save_path: Optional[str] = None):
        """Построение графиков распределения основных характеристик"""
        if self.stats_df is None:
            self.analyze_dataset()

        # Создаем сетку графиков
        fig, axes = plt.subplots(2, 3, figsize=(25, 16))
        fig.suptitle('Распределение характеристик аудиофайлов', fontsize=16)

        # Длительность
        sns.histplot(data=self.stats_df, x='duration', bins=50, ax=axes[0,0])
        axes[0,0].set_title('Распределение длительности')
        axes[0,0].set_xlabel('Длительность (секунды)')

        # RMS энергия
        sns.histplot(data=self.stats_df, x='rms_energy', bins=50, ax=axes[0,1])
        axes[0,1].set_title('Распределение RMS энергии')
        axes[0,1].set_xlabel('RMS энергия')

        # Процент тишины
        sns.histplot(data=self.stats_df, x='silence_percentage', bins=50, ax=axes[0,2])
        axes[0,2].set_title('Распределение процента тишины')
        axes[0,2].set_xlabel('Процент тишины')

        # Zero crossing rate
        sns.histplot(data=self.stats_df, x='zero_crossing_rate', bins=50, ax=axes[1,0])
        axes[1,0].set_title('Распределение Zero Crossing Rate')
        axes[1,0].set_xlabel('Zero Crossing Rate')

        # Спектральный центроид
        sns.histplot(data=self.stats_df, x='spectral_centroid', bins=50, ax=axes[1,1])
        axes[1,1].set_title('Распределение спектрального центроида')
        axes[1,1].set_xlabel('Спектральный центроид')

        # Sample rate (bar plot)
        sample_rates = self.stats_df['sample_rate'].value_counts()
        sample_rates.plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Распределение частоты дискретизации')
        axes[1,2].set_xlabel('Sample Rate (Hz)')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """Построение матрицы корреляций между характеристиками"""
        if self.stats_df is None:
            self.analyze_dataset()

        # Вычисляем корреляции
        corr = self.stats_df.corr()

        # Создаем тепловую карту
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Корреляция между характеристиками аудио')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_duration_vs_feature(self, feature: str, save_path: Optional[str] = None):
        """Построение графика зависимости характеристики от длительности"""
        if self.stats_df is None:
            self.analyze_dataset()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.stats_df, x='duration', y=feature)
        plt.title(f'Зависимость {feature} от длительности')
        plt.xlabel('Длительность (секунды)')
        plt.ylabel(feature)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()



# Инициализация анализатора
analyzer = AudioDatasetAnalyzer('AI_by_Francois_Chollet/audio_ml/train/')
analyzer.load_jsonl('AI_by_Francois_Chollet/audio_ml/train/10hours.jsonl')

# Анализ датасета
analyzer.analyze_dataset()

# Построение распределений всех характеристик
analyzer.plot_distributions('distributions.png')

# Построение матрицы корреляций
analyzer.plot_correlation_matrix('correlations.png')

# Построение графика зависимости RMS энергии от длительности
analyzer.plot_duration_vs_feature('rms_energy', 'duration_vs_rms.png')

filtered = analyzer.filter_dataset(1, 10, 0.01, 80)
print(filtered)