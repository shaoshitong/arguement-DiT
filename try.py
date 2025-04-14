import matplotlib.pyplot as plt

# Prepare the six lists as requested: random_mu, dc_mu, random_sigma, dc_sigma, random_overal, dc_overal
fid_mu_random = [102.52987727294544, 45.893179702993066, 34.13368751473661, 25.433631481866485, 
             19.810743029952757, 15.893837869765436, 12.439949122326578, 10.240969697374265, 
             8.361258485440924, 7.1271124338010985]

fid_mu_dc = [98.64821507931862, 39.88746246593957, 29.11899923819022, 21.60240755440338, 
         15.857268004461812, 12.159677070162196, 9.484210477518673, 7.937959517423394, 
         6.4331961797618895, 5.4981362985207465]

fid_sigma_random = [132.31252843807334, 49.524796831753235, 32.3133182757519, 26.020549113528602, 
                22.13676759938636, 19.929892437446426, 17.666830060938878, 16.111546907456045, 
                14.809094076309862, 14.002710000861043]

fid_sigma_dc = [129.63268063326464, 47.86334490562666, 34.96324735962867, 28.621885007708556, 
            24.64744033061703, 21.571434219423622, 19.338512258315916, 18.03270609356548, 
            16.796376156431734, 16.01449610750211]

fid_overal_random = [234.84240571101878, 95.4179765347463, 66.4470057904885, 51.45418059539509, 
                 41.94751062933912, 35.82373030721186, 30.106779183265456, 26.35251660483031, 
                 23.170352561750786, 21.12982243466214]

fid_overal_dc = [228.28089571258326, 87.75080737156622, 64.08224659781888, 50.22429256211194, 
             40.504708335078845, 33.73111128958582, 28.82272273583459, 25.970665610988874, 
             23.229572336193623, 21.51263240602286]

mu_x = []
for i in range(10):
    mu_x.append(fid_mu_random[i] - fid_mu_dc[i])
sigma_x = []
for i in range(10):
    sigma_x.append(fid_sigma_dc[i] - fid_sigma_random[i])
overal_x = []
for i in range(10):
    overal_x.append(fid_overal_random[i] - fid_overal_dc[i])
# Prepare the y-axis limits



y_min = -5
y_max = 8.5

x_values = [2,4,6,8,10,12,14,16,18,20]
# Create a figure with 2 rows and 2 columns for the original value plots
# 创建一个包含3个子图的2行1列布局
# 创建一个包含3个子图的2行2列布局

# 创建一个包含3个子图的1行2列 + 1行1列布局
# 创建一个包含3个子图的2行2列布局，其中第二个位置留空
# 创建一个包含3个子图的2行2列布局
fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(14, 12))  # 创建2行2列的布局，空白位置用 _ 填充

# 绘制 FiD_mu（ax1）
# ax1.plot(x_values, fid_mu_random, label='random', marker='o')
# ax1.plot(x_values, fid_mu_dc, label='dc', marker='o')
ax1.plot(x_values,mu_x, label='dc', marker='o')

ax1.set_title('FiD_mu for Random and DC')
ax1.set_xlabel('x (6, 8, ..., 20)')
ax1.set_ylabel('FiD_mu')
ax1.set_xticks(x_values)
ax1.set_ylim([y_min, y_max])  # 设置统一的y轴范围
ax1.grid(True)
ax1.legend()

# 绘制 FiD_sigma（ax2）
# ax2.plot(x_values, fid_sigma_random, label='random', marker='o')
# ax2.plot(x_values, fid_sigma_dc, label='dc', marker='o')
ax2.plot(x_values, sigma_x, label='dc', marker='o')

ax2.set_title('FiD_sigma for Random and DC')
ax2.set_xlabel('x (6, 8, ..., 20)')
ax2.set_ylabel('FiD_sigma')
ax2.set_xticks(x_values)
ax2.set_ylim([y_min, y_max])  # 设置统一的y轴范围
ax2.grid(True)
ax2.legend()

# 绘制 FiD_overal（ax3）
# ax3.plot(x_values, fid_overal_random, label='random', marker='o')
# ax3.plot(x_values, fid_overal_dc, label='dc', marker='o')
ax3.plot(x_values, overal_x, label='dc', marker='o')

ax3.set_title('FiD_overal for Random and DC')
ax3.set_xlabel('x (6, 8, ..., 20)')
ax3.set_ylabel('FiD_overal')
ax3.set_xticks(x_values)
ax3.grid(True)
ax3.legend()

# 调整布局，防止子图重叠
fig.tight_layout()

# 保存包含所有三个子图的最终图像
plt.savefig('fiD_plots_combined.png', dpi=300)