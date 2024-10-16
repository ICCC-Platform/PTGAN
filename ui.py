from pathlib import Path
import sys
import os.path as osp
from PIL import Image
import gradio as gr
import torch
from metrics import metrices_map
import os
from PIL import Image, ImageDraw, ImageFont
from config import cfg
from dataset import get_transform
from models import TransReIDBase_Inference
from tools.jsonio import read_json


class ReID_Pipline():
    
    def __init__(self, cfg, gallery_precalculate_path:os.PathLike) -> None:
        
        self.dev = torch.device(f"cuda:{cfg.MODEL.DEVICE_ID}")

        self.model = self._build_model(cfg=cfg).to(device=self.dev)
        
        self.gallery_feature:torch.Tensor = torch.load(
            Path(gallery_precalculate_path)/"gallery_reid_features.pt", 
            map_location="cpu"
        ).to(device=self.dev)

        self.gallery_item = read_json(Path(gallery_precalculate_path)/"gallery_items.json")
        self.T = get_transform(
            cfg.INPUT.SIZE_TEST, 
            {'mean':cfg.INPUT.PIXEL_MEAN, 'std':cfg.INPUT.PIXEL_STD}
        )
    
    def _build_model(self, cfg) -> TransReIDBase_Inference:
        M = TransReIDBase_Inference(cfg=cfg)
        M.eval()
        return M

    def __call__(self, x:Image.Image, topk:int=5, distance_method:str="cosine", **kwargs) ->tuple[list[os.PathLike], list[float]]:
        
        xi:torch.Tensor = self.T(x)
        if xi.ndim == 3:
            xi = xi.unsqueeze(0) 

        dist_matrix:torch.Tensor = metrices_map[distance_method](
            self.model(xi.to(self.dev)), 
            self.gallery_feature, 
            **kwargs
        )

        sim = dist_matrix.argsort(dim=1)
        target = sim[0, :topk]

        return [self.gallery_item[i] for i in target], dist_matrix[0, target].tolist()

reid_pipline:ReID_Pipline = None

class WebUI():
    
    def __init__(
            self, query_show_size:tuple[int]=(200, 200), 
            gallery_img_root:os.PathLike = Path("VehicleData")/"gallery"
        ) -> None:
         match sys.platform:
            case  "win32":
                self.USE_FONT = "C:\\Windows\\Fonts\\times.ttf" 
            case "linux":
                self.USE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        self.g_root = gallery_img_root
        self.q_img = None
        self.query_show_size = query_show_size

        self.metrics_args_map = {
            're_ranking':{
                'k1':50, 
                'k2':15, 
                'lambda_value':0.3,
                'pbar':  gr.Progress()

            }
        }

    def run_app(self):
        with gr.Blocks(
            theme=gr.Theme(text_size=gr.themes.sizes.text_lg),
            title="ReID Web UI", fill_height=True
        ) as demo:
            
            title = gr.Label("ReID Demo", show_label=False)
            with gr.Column(min_width=self.query_show_size[1]):
                with gr.Row(equal_height=False):
            
                    with gr.Column(min_width=self.query_show_size[1]):
                        self.q_img = gr.Image(label='Query', type='pil', show_download_button=False)
                        
                    with gr.Column(min_width=50):
                        topk_input = gr.Textbox(label="Top K", min_width=50)
                        metrices_bar = gr.Dropdown(label="Distance Type:",choices=["l2", "cosine", "re_ranking"], scale=4)
                        reid_bnt = gr.Button("Match")
            
                with gr.Row(equal_height=False):
                    result_imgs1 = [
                        gr.Image(
                            label=f"top{_+1}",
                            width=self.query_show_size[1], 
                            min_width=self.query_show_size[1], 
                            show_download_button=False
                        )
                        for _ in range(5)
                    ]
                with gr.Row(equal_height=False):
                    label1 = [gr.Label(f"", show_label=False) for _ in range(5)]
                with gr.Row(equal_height=False):
                    result_imgs2 = [
                        gr.Image(
                            label=f"top{_+6}",width=self.query_show_size[1], 
                            min_width=self.query_show_size[1], 
                            show_download_button=False
                        ) 
                        for _ in range(5)
                    ]
                with gr.Row(equal_height=False):
                    label2 = [gr.Label(f"",  show_label=False) for _ in range(5)]
        
                reid_bnt.click(
                    self.Inference, inputs=[topk_input, self.q_img, metrices_bar], 
                    outputs=result_imgs1+result_imgs2+label1+label2
                )
            
            demo.launch(inbrowser=True)
            demo.close()
        
    def Inference(self, topk:str, img:Image.Image, metrics:str):
        topK = int(topk)
        additional_args = self.metrics_args_map.get(metrics, {})
        candidates, dist = reid_pipline(
            img.convert("RGB"), topK, 
            distance_method=metrics, **additional_args
        )
        candidates_imgs = [
            self.adding_text_card(image_path = (self.g_root/ci),txt=f"{dist[i]:.3f}")
            for i, ci in enumerate(candidates) if i < 10
        ]
        
        candidates = list(f"CarID:{osp.split(c)[-1][:4]}\nSeqNum:{osp.split(c)[-1][11:18]}" for c in candidates)
        
        if 10 - topK > 0:
            candidates_imgs += [Image.new("RGB", self.query_show_size, color="white") for _ in range(10-topK)]
            candidates += [""]*(10-topK)
       
        return candidates_imgs + candidates

    def adding_text_card(self, image_path:os.PathLike, txt:str) -> Image.Image:
        
        image = Image.open(image_path).resize((200,200))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.USE_FONT, size=16)
        text_size = font.getbbox(txt)
        box_color =  (255, 165, 0)
        # Draw the text box
        draw.rectangle(
            xy=[(image.width - text_size[2] - 1 - 10 , 0), 
                (image.width-1, text_size[3] + 10 )], 
            fill=box_color
        )

        draw.text((image.width - text_size[2] - 5 ,  (text_size[3] + 10)/2 - 10) , txt, fill=(0,0,0), font=font)

        return image

if __name__ == "__main__":
    
    cfg.merge_from_file(Path("config")/"transreid_256_veri_gan.yml")
    cfg.freeze()
    
    reid_pipline = ReID_Pipline(
        cfg=cfg, 
        gallery_precalculate_path=Path("precalculated")/"gallery"
    )

    ui = WebUI()
    ui.run_app()
