"""
Point-E Agent for Blazer AI - Fixed for PyTorch 2.9.1 compatibility
Generates 3D point clouds from text prompts using OpenAI's Point-E model.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add Point-E to path
POINT_E_PATH = r"C:\Users\arunk\Downloads\blazer\point-e"
if POINT_E_PATH not in sys.path:
    sys.path.insert(0, POINT_E_PATH)

# Check for required packages
POINT_E_AVAILABLE = False
try:
    import torch
    import trimesh
    
    # Point-E imports
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    
    POINT_E_AVAILABLE = True
    print("[POINT-E] Point-E imports successful!")
except ImportError as e:
    print(f"[POINT-E] Import error: {e}")
    POINT_E_AVAILABLE = False


class PointEAgent:
    """Agent for generating 3D models using Point-E"""
    
    def __init__(self):
        self.available = False
        self.models_loaded = False
        
        if not POINT_E_AVAILABLE:
            print("[POINT-E] Point-E not available. Skipping initialization.")
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[POINT-E] Using device: {self.device}")
        
        self.available = True
        self.base_model = None
        self.upsampler_model = None
    
    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if self.models_loaded:
            return True
            
        try:
            print("[POINT-E] Loading models (first time, may take 2-3 minutes)...")
            
            # Load base model - need to create model then load state dict
            self.base_name = 'base40M-textvec'
            print(f"[POINT-E] Loading {self.base_name}...")
            self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], device=self.device)
            self.base_model.load_state_dict(load_checkpoint(self.base_name, device=self.device))
            self.base_model.eval()
            
            # Load upsampler
            self.upsampler_name = 'upsample'
            print(f"[POINT-E] Loading {self.upsampler_name}...")
            self.upsampler_model = model_from_config(MODEL_CONFIGS[self.upsampler_name], device=self.device)
            self.upsampler_model.load_state_dict(load_checkpoint(self.upsampler_name, device=self.device))
            self.upsampler_model.eval()
            
            print("[POINT-E] Models loaded successfully!")
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"[POINT-E] Error loading models: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
            return False
    
    def generate_point_cloud(self, prompt, guidance_scale=3.0):
        """Generate a 3D point cloud from a text prompt"""
        if not self.available:
            raise Exception("Point-E is not available")
        
        if not self._ensure_models_loaded():
            raise Exception("Failed to load Point-E models")
        
        print(f"[POINT-E] Generating point cloud for: '{prompt}'")
        
        try:
            # Create diffusion configs
            base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
            upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.upsampler_name])
            
            # Create sampler
            sampler = PointCloudSampler(
                device=self.device,
                models=[self.base_model, self.upsampler_model],
                diffusions=[base_diffusion, upsampler_diffusion],
                num_points=[1024, 4096 - 1024],
                aux_channels=['R', 'G', 'B'],
                guidance_scale=[guidance_scale, 0.0],
                use_karras=[True, True],
                karras_steps=[64, 64],
                sigma_min=[1e-3, 1e-3],
                sigma_max=[120.0, 160.0],
                s_churn=[3.0, 0.0],
            )
            
            # Generate with progress
            print("[POINT-E] Starting generation (this takes ~2-5 minutes on CPU)...")
            samples = None
            for i, x in enumerate(sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(texts=[prompt]),
            )):
                samples = x
                if i % 20 == 0:
                    print(f"[POINT-E] Step {i}...")
            
            # Extract point cloud
            pc = sampler.output_to_point_clouds(samples)[0]
            
            # Combine coordinates and colors
            coords = pc.coords
            colors = np.stack([
                pc.channels['R'], 
                pc.channels['G'], 
                pc.channels['B']
            ], axis=-1)
            
            point_cloud = np.concatenate([coords, colors], axis=-1)
            
            print(f"[POINT-E] Generated point cloud with {len(point_cloud)} points")
            return point_cloud
            
        except Exception as e:
            print(f"[POINT-E] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_point_cloud(self, point_cloud, filepath):
        """Save point cloud to PLY file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        coords = point_cloud[:, :3]
        colors = (point_cloud[:, 3:6] * 255).astype(np.uint8)
        
        # Create trimesh point cloud
        pc_mesh = trimesh.PointCloud(vertices=coords, colors=colors)
        pc_mesh.export(str(filepath))
        
        print(f"[POINT-E] Saved to: {filepath}")
        return str(filepath)
    
    def generate_3d(self, prompt, save_path=None, format='ply'):
        """Main method to generate 3D model from text"""
        try:
            print(f"[POINT-E] Starting 3D generation for: '{prompt}'")
            
            # Generate point cloud
            point_cloud = self.generate_point_cloud(prompt)
            
            # Save if path provided
            file_path = None
            if save_path:
                if not save_path.endswith(f'.{format}'):
                    save_path = f"{save_path}.{format}"
                file_path = self.save_point_cloud(point_cloud, save_path)
            
            return {
                'success': True,
                'prompt': prompt,
                'file_path': file_path,
                'point_count': len(point_cloud),
                'format': format
            }
            
        except Exception as e:
            print(f"[POINT-E] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# Global instance
_point_e_agent = None

def get_point_e_agent():
    """Get or create Point-E agent instance"""
    global _point_e_agent
    if _point_e_agent is None:
        _point_e_agent = PointEAgent()
    return _point_e_agent
