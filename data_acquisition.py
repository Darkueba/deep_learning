"""
MODULE 1: DATA ACQUISITION & PREPROCESSING
Uses Sentinel Hub API to download Landsat 8 imagery
Includes visualization and basic preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox
from sentinelhub import DataCollection
import json
from tqdm import tqdm

try:
    import config
except ImportError:
    print("Warning: config.py not found")
    raise


class LandsatDataAcquisition:
    """
    Downloads Landsat-8 imagery using Sentinel Hub API
    """
    
    def __init__(self):
        """Initialize Sentinel Hub configuration"""
        print("\n" + "="*70)
        print("LANDSAT DATA ACQUISITION MODULE (Sentinel Hub)")
        print("="*70)
        
        # Setup Sentinel Hub config
        self.config = SHConfig()
        self.config.sh_client_id = config.SH_CLIENT_ID
        self.config.sh_client_secret = config.SH_CLIENT_SECRET
        
        # Create output directories
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        
        # Bounding box
        self.bbox = BBox(bbox=config.BBOX, crs=CRS.WGS84)
        
        print(f"\n✓ Sentinel Hub initialized")
        print(f"  Study Area: {config.STUDY_AREA_NAME}")
        print(f"  Bounding Box: {config.BBOX}")
        print(f"  Time Range: {config.START_DATE} to {config.END_DATE}")
    
    def create_evalscript(self):
        """
        Create Evalscript for Sentinel Hub to select specific bands
        
        Returns:
        --------
        str : Evalscript code
        """
        # Create evalscript that selects all requested bands
        bands_str = ", ".join([f'"{band}"' for band in config.DOWNLOAD_BANDS])
        
        evalscript = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{bands_str}],
    output: {{ bands: {len(config.DOWNLOAD_BANDS)} }}
  }};
}}
function evaluatePixel(sample) {{
  return [{', '.join([f'sample.{band}' for band in config.DOWNLOAD_BANDS])}];
}}
"""
        return evalscript.strip()
    
    def download_imagery(self):
        """
        Download Landsat 8 imagery from Sentinel Hub
        
        Returns:
        --------
        numpy.ndarray : Downloaded imagery (height, width, bands)
        dict : Metadata
        """
        print(f"\n{'='*70}")
        print("DOWNLOADING LANDSAT-8 IMAGERY")
        print(f"{'='*70}")
        
        # Create evalscript
        evalscript = self.create_evalscript()
        
        # Create request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.LANDSAT_OT_L1,
                    time_interval=(config.START_DATE, config.END_DATE),
                    maxcc=config.MAX_CLOUD_COVER
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=self.bbox,
            size=config.IMAGE_SIZE,
            config=self.config
        )
        
        try:
            print("\n⏳ Downloading from Sentinel Hub...")
            data = request.get_data()[0]  # First time step
            
            print(f"✓ Download successful!")
            print(f"  Image shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Min value: {data.min()}, Max value: {data.max()}")
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'shape': data.shape,
                'bbox': config.BBOX,
                'bands': config.DOWNLOAD_BANDS,
                'spatial_resolution': config.SPATIAL_RESOLUTION,
                'source': 'Sentinel Hub / Landsat OT L1'
            }
            
            return data, metadata
            
        except Exception as e:
            print(f"✗ Download failed: {str(e)}")
            print("\nTroubleshooting:")
            print("  1. Check Sentinel Hub credentials in config.py")
            print("  2. Verify internet connection")
            print("  3. Check that bounding box is valid")
            raise
    
    def save_imagery(self, data, metadata):
        """
        Save downloaded imagery and metadata
        
        Parameters:
        -----------
        data : numpy.ndarray
            Imagery array
        metadata : dict
            Metadata information
        """
        # Save raw data
        data_path = os.path.join(config.RAW_DATA_DIR, 'landsat_raw.npy')
        np.save(data_path, data)
        print(f"\n✓ Raw imagery saved: {data_path}")
        
        # Save individual bands
        print("\nSaving individual bands:")
        for i, band in enumerate(config.DOWNLOAD_BANDS):
            band_path = os.path.join(config.RAW_DATA_DIR, f'{band}.npy')
            np.save(band_path, data[:, :, i])
            print(f"  ✓ {band}: {band_path}")
        
        # Save metadata
        metadata_path = os.path.join(config.RAW_DATA_DIR, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")
        
        return data_path
    
    def visualize_imagery(self, data):
        """
        Visualize downloaded imagery
        
        Parameters:
        -----------
        data : numpy.ndarray
            Imagery array
        """
        print(f"\n{'='*70}")
        print("VISUALIZING IMAGERY")
        print(f"{'='*70}")
        
        # Band indices (assuming order: B04, B03, B02, B05, B06, B07)
        # which corresponds to: R, G, B, NIR, SWIR1, SWIR2
        red_idx = 0    # B04
        green_idx = 1  # B03
        blue_idx = 2   # B02
        nir_idx = 3    # B05
        swir1_idx = 4  # B06
        swir2_idx = 5  # B07
        
        # Normalize for visualization (simple min-max scaling)
        def normalize(band):
            band = band.astype(float)
            return (band - band.min()) / (band.max() - band.min() + 1e-8)
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Landsat-8 Imagery - {config.STUDY_AREA_NAME}', fontsize=16)
        
        # RGB Composite
        rgb = np.stack([
            normalize(data[:, :, red_idx]),
            normalize(data[:, :, green_idx]),
            normalize(data[:, :, blue_idx])
        ], axis=2)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB (True Color)')
        axes[0, 0].axis('off')
        
        # False Color Composite (NIR-Red-Green for vegetation)
        fcc = np.stack([
            normalize(data[:, :, nir_idx]),
            normalize(data[:, :, red_idx]),
            normalize(data[:, :, green_idx])
        ], axis=2)
        axes[0, 1].imshow(fcc)
        axes[0, 1].set_title('False Color (Vegetation)')
        axes[0, 1].axis('off')
        
        # NIR Band
        axes[0, 2].imshow(normalize(data[:, :, nir_idx]), cmap='Greens')
        axes[0, 2].set_title('NIR Band (B05)')
        axes[0, 2].axis('off')
        
        # Red Band
        axes[1, 0].imshow(normalize(data[:, :, red_idx]), cmap='Reds')
        axes[1, 0].set_title('Red Band (B04)')
        axes[1, 0].axis('off')
        
        # SWIR1 Band
        axes[1, 1].imshow(normalize(data[:, :, swir1_idx]), cmap='Blues')
        axes[1, 1].set_title('SWIR1 Band (B06)')
        axes[1, 1].axis('off')
        
        # Histogram of Red Band
        axes[1, 2].hist(data[:, :, red_idx].flatten(), bins=100, color='red', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Red Band Histogram')
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(config.FIGURES_DIR, '01_raw_imagery_visualization.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {fig_path}")
        plt.close()
    
    def compute_statistics(self, data):
        """
        Compute and display statistics of downloaded imagery
        
        Parameters:
        -----------
        data : numpy.ndarray
            Imagery array
        
        Returns:
        --------
        dict : Statistics for each band
        """
        print(f"\n{'='*70}")
        print("DATA STATISTICS")
        print(f"{'='*70}")
        
        stats = {}
        print(f"\n{'Band':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 50)
        
        for i, band_name in enumerate(config.DOWNLOAD_BANDS):
            band_data = data[:, :, i]
            stats[band_name] = {
                'mean': float(band_data.mean()),
                'std': float(band_data.std()),
                'min': float(band_data.min()),
                'max': float(band_data.max()),
                'median': float(np.median(band_data))
            }
            
            print(f"{band_name:<10} {stats[band_name]['mean']:<12.2f} "
                  f"{stats[band_name]['std']:<12.2f} "
                  f"{stats[band_name]['min']:<12.2f} "
                  f"{stats[band_name]['max']:<12.2f}")
        
        return stats
    
    def run(self):
        """
        Execute complete data acquisition pipeline
        
        Returns:
        --------
        tuple : (imagery_data, metadata, statistics)
        """
        try:
            # Download imagery
            data, metadata = self.download_imagery()
            
            # Save data
            self.save_imagery(data, metadata)
            
            # Visualize
            self.visualize_imagery(data)
            
            # Compute statistics
            stats = self.compute_statistics(data)
            
            print(f"\n{'='*70}")
            print("✓ DATA ACQUISITION COMPLETE")
            print(f"{'='*70}")
            
            return data, metadata, stats
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution"""
    # Initialize and run
    acq = LandsatDataAcquisition()
    data, metadata, stats = acq.run()
    
    print(f"\nData shape: {data.shape}")
    print(f"Ready for feature extraction!")


if __name__ == "__main__":
    main()
