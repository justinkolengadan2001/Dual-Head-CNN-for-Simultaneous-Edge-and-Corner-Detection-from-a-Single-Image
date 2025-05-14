Project Name: Dual-Head CNN for Simultaneous Edge and Corner Detection from a Single Image

-- Project presents a dual-head convolutional neural network (CNN) that simultaneously detects edges and corners. 
-- Explored two model backbones — UNetDualHead and ResNetDualHeadSkip and conducted extensive ablations with different loss functions [BCE, Weighted BCE, Focal Loss], optimizers [Adam, AdamW, SGD with momentum], and activations [ReLU, Leaky ReLU, Mish].

------------------------------------------------------------------------------------------------------------------------------------------------------------

[1] Setup Instructions
1. Install Dependencies
-- Run the following command to install all required packages:
	pip install -r requirements.txt

[2] Data Preparation
1. Place jcsmr.jpg inside ../data/original/.
2. Run the cropping and filtering notebook at ../notebooks/preprocessing.ipynb to generate ground truth labels:
	RGB patches stored in ../data/filtered_crops/rgb/
	Edge labels (Canny) stored in ../data/filtered_crops/canny/
	Corner labels (Harris) stored in ../data/filtered_crops/harris/

[3] Training the Model
1. Train both U-Net and ResNet models:
	python train.py

	## Note: Currently best model setup is set for training in ../model.py. 
	         Other model architectures available in ../notebooks/PyTorch_Model_Training_Kaggle.ipynb

	Trained weights will be saved in ../models/
	Metrics will be plotted and saved in ../plots/
	Evaluation metrics will be exported to CSV in ../metrics/
	Random Single Image Prediction will be plotted and saved in ../predictions/

[4] Inference (Testing)
1. Test a trained model on a custom image:
	python inference.py --model resnetdualhead --image test.png
   For U-Net:
	python inference.py --model unetdualhead --image test.png
	
	## Note: (Optional) --weights argument can be given to test other variants of the model. Currently weights set to best model in each category.
	Example Usage: python inference.py --model unetdualhead --image test.png --weights ../models/unet_dual_head_best_adamW_bce.pth

	Input image should be resized to 512 × 512
	Outputs will be saved in ../outputs/ 