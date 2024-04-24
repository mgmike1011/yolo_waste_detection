import argparse

from ultralytics import YOLO

from utilities.parsing_vaildator import dir_path, file_path


def main(input_data: file_path, model_path: file_path, output_folder: dir_path, run_name: str, resume_run: bool):
    # Hyperparameters - https://docs.ultralytics.com/modes/train/#train-settings
    data_ = input_data
    model_ = model_path
    epochs_ = 100
    time_ = None
    patience_ = 100
    batch_ = 16
    imgsz_ = 640
    save_ = True
    save_period_ = -1
    cache_ = False
    device_ = "0"
    workers_ = 8
    project_ = output_folder
    name_ = run_name
    exist_ok_ = False
    pretrained_ = True
    optimizer_ = "auto"
    verbose_ = False
    seed_ = 0
    deterministic_ = True
    single_cls_ = False
    rect_ = False
    cos_lr_ = False
    close_mosaic_ = 10
    resume_ = resume_run
    amp_ = True
    fraction_ = 1.0
    profile_ = False
    freeze_ = None
    lr0_ = 0.01
    lrf_ = 0.01
    momentum_ = 0.937
    weight_decay_ = 0.0005
    warmup_epochs_ = 3.0
    warmup_momentum_ = 0.8
    warmup_bias_lr_ = 0.1
    box_ = 7.5
    cls_ = 0.5
    dfl_ = 1.5
    pose_ = 12.0
    kobj_ = 2.0
    label_smoothing_ = 0.0
    nbs_ = 64
    overlap_mask_ = True
    mask_ratio_ = 4
    dropout_ = 0.0
    val_ = True
    plots_ = False

    # Load model and train
    model = YOLO(model_)
    print("--- Train start ---")
    model.train(data=data_, epochs=epochs_, time=time_, patience=patience_, batch=batch_, imgsz=imgsz_, save=save_,
                save_period=save_period_, cache=cache_, device=device_, workers=workers_, project=project_, name=name_,
                exist_ok=exist_ok_, pretrained=pretrained_, optimizer=optimizer_, verbose=verbose_, seed=seed_,
                deterministic=deterministic_, single_cls=single_cls_, rect=rect_, cos_lr=cos_lr_,
                close_mosaic=close_mosaic_, resume=resume_, amp=amp_, fraction=fraction_, profile=profile_,
                freeze=freeze_, lr0=lr0_, lrf=lrf_, momentum=momentum_, weight_decay=weight_decay_,
                warmup_epochs=warmup_epochs_, warmup_momentum=warmup_momentum_, warmup_bias_lr=warmup_bias_lr_,
                box=box_, cls=cls_, dfl=dfl_, pose=pose_, kobj=kobj_, label_smoothing=label_smoothing_, nbs=nbs_,
                overlap_mask=overlap_mask_, mask_ratio=mask_ratio_, dropout=dropout_, val=val_, plots=plots_)
    print("--- Train end ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO train model with given input data.")

    parser.add_argument("--data", type=file_path, help="Input data YAML file.", required=True)
    parser.add_argument("--model", type=file_path, help="Path to a model.", required=True)
    parser.add_argument("--output", type=dir_path, help="Path to output directory.", required=True)
    parser.add_argument("--name", type=dir_path, help="Run name.", required=True)
    parser.add_argument("--resume", type=bool, help="Resumes training from the last saved checkpoint.",
                        default=False, required=False)

    args = parser.parse_args()

    main(input_data=args.data, model_path=args.model, output_folder=args.output, run_name=args.name,
         resume_run=args.resume)
