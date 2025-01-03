import pandas as pd
import matplotlib.pyplot as plt

def plot_training_loss(file_path, output_path):
    # 读取表格数据
    data = pd.read_excel(file_path)

    # 检查是否包含需要的列
    required_columns = ["Epoch", "Supervised Loss", "Unsupervised Loss"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # 提取数据
    epochs = data["Epoch"]
    supervised_loss = data["Supervised Loss"]
    unsupervised_loss = data["Unsupervised Loss"]

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, supervised_loss)
    #plt.plot(epochs, 0.1* unsupervised_loss, label="Unsupervised Loss")

    # 添加图例和标签
    plt.title("Training Loss vs Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    plt.show()

# 示例用法
# 文件路径替换为实际的文件路径，例如 'data/losses.xlsx'
file_path = "training_loss_output.xlsx"
output_path = "training_loss_plot.png"
plot_training_loss(file_path, output_path)
