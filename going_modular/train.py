if __name__ == "__main__":
    import os
    import torch
    from torchvision import transforms
    import data_setup, engine, model_builder, utils

    NUM_EPOCHS=5
    BATCH_SIZE=32
    HIDDEN_UNITS=10
    LEARNING_RATE=0.001

    train_data="data/pizza_steak_sushi/train"
    test_data="data/pizza_steak_sushi/test"

    device="cuda" if torch.cuda.is_available() else "cpu"

    data_transform=transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(
        train_dir=train_data,
        test_dir=test_data,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model=model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)).to(device)

    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(params=model.parameters(),
                            lr=LEARNING_RATE)

    from timeit import default_timer as timer 
    start_time = timer()

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    utils.save_model(model=model,
                target_dir="models",
                model_name="05_going_modular_script_tinyvgg_model.pth")
