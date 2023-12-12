from baseline_train import *
import baseline_train

def main():
    
   model_G = torch.load("./Sat2MapGen_v1.5.pth")
   model_D = torch.load("./Sat2MapDisc_v1.5.pth")
   
   model_G = torch.load("./Map2SatGen_v1.5.pth")
   model_D = torch.load("./Map2SatDisc_v1.5.pth")
   
if __name__ == "__main__":
    main()

