#sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name InceptionResNetV2 --weights imagenet"
# sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name InceptionResNetV2 --weights radimagenet"

 #sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name ResNet50 --weights imagenet"
# sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name ResNet50 --weights radimagenet"

#sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name InceptionV3 --weights imagenet"
# sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name InceptionV3 --weights radimagenet"

#sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name DenseNet121 --weights imagenet"
# sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name DenseNet121 --weights radimagenet"

#sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name Xception"
#sbatch -N 1 -n 1 -c 1 -p volta-gpu --mem=16g -t 168:00:00 --qos gpu_access --gres=gpu:1 --mail-user=kevin.chen@unchealth.unc.edu --wrap="python3 rin.py --model_name BiT"
echo "hello"