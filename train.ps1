./venv310/Scripts/activate

foreach ($color_space in @('rgb', 'hsv', 'bgr', 'ycrcb'))
{
    foreach ($kernel_size in @(3, 5))
    {
        $command = "./train.py --loss mse --kernel_size $($kernel_size) --color_space $($color_space)",
        "--batch_size 4 --train_data ./LOL-v2/Synthetic/Train ./LOL-v2/Real_captured/Train ",
        "--test_data ./LOL-v2/Synthetic/Test ./LOL-v2/Real_captured/Test",
        "--device cuda:0 --epochs 300"
        echo "`n$command"
        Start-Process python.exe $command -Wait -NoNewWindow
    }
}