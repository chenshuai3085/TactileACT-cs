#!/usr/bin/env python3
"""
专利二（完整版）：
一种面向人形机器人螺栓装配的多模态感知智能系统及方法
——覆盖从正常执行到异常处理的完整智能装配闭环
"""

from docx import Document
from docx.shared import Pt as DocPt, Cm as DocCm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os

FIG_DIR = '/home/chenshuai/Project/TactileACT-cs/paper/patent/figures'


def set_cn_font(run, name='宋体', size=DocPt(12)):
    run.font.name = name
    run.font.size = size
    r = run._element
    rPr = r.find(qn('w:rPr'))
    if rPr is None:
        rPr = r.makeelement(qn('w:rPr'), {})
        r.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = rPr.makeelement(qn('w:rFonts'), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), name)


def P(doc, text, bold=False, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = DocCm(0.74)
    run = p.add_run(text)
    run.bold = bold
    set_cn_font(run, '宋体', DocPt(12))
    return p


def H(doc, text, level=2):
    return doc.add_heading(text, level=level)


def add_figure(doc, fig_path, caption):
    """插入配图并添加居中图注"""
    if os.path.exists(fig_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(fig_path, width=DocCm(14))
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_cap = p_cap.add_run(caption)
    run_cap.bold = True
    set_cn_font(run_cap, '宋体', DocPt(10.5))
    doc.add_paragraph()


def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = DocCm(2.54)
        s.bottom_margin = DocCm(2.54)
        s.left_margin = DocCm(3.17)
        s.right_margin = DocCm(3.17)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('技术交底书')
    r.bold = True
    set_cn_font(r, '黑体', DocPt(18))
    doc.add_paragraph()

    P(doc, '发明名称：一种面向人形机器人螺栓装配的多模态感知智能系统及方法', bold=True, indent=False)
    doc.add_paragraph()

    # ============ 验证性 ============
    H(doc, '验证性')
    P(doc, '容易验证。', bold=True, indent=False)
    P(doc,
      '取证方式：(1) 在汽车总装车间部署本系统后，观察人形机器人是否能在无人干预下完成'
      '螺栓"抓取→对准→旋拧→力矩达标"的完整装配流程（可从控制日志中提取每个阶段的时序数据验证）；'
      '(2) 人为制造装配异常（如将螺栓故意倾斜3°放入螺纹孔），观察系统是否能自动检出、'
      '输出自然语言诊断结论、并自主执行修正动作；'
      '(3) 检查系统是否包含"安全分级"逻辑——低风险修正直接执行、中风险需操作员默认确认、'
      '高风险强制转人工；'
      '(4) 检查系统是否在每次装配后自动记录多模态传感数据和诊断结果，'
      '用于经验积累和模型持续改进。')

    P(doc, '缩略语说明：', bold=True, indent=False)
    P(doc,
      'VLT: Vision-Language-Tactile, 视觉-语言-触觉; '
      'LLM: Large Language Model, 大语言模型; '
      'VLA: Vision-Language-Action, 视觉-语言-动作模型; '
      'F/T: Force/Torque, 力/力矩; '
      'RAG: Retrieval-Augmented Generation, 检索增强生成; '
      'CoT: Chain-of-Thought, 思维链推理; '
      'LoRA: Low-Rank Adaptation, 低秩适配微调; '
      'VQ: Vector Quantization, 向量量化')

    # ============ 1.1 技术领域 ============
    H(doc, '1.1 技术领域')
    P(doc, '1、本技术方案所属的技术领域', bold=True, indent=False)
    P(doc,
      '本发明属于智能制造与人形机器人自主控制技术领域，具体涉及一种'
      '面向人形机器人在工业产线执行螺栓/螺柱装配任务时，'
      '覆盖"正常装配执行—实时异常监测—异常自主诊断—分级修正"完整闭环的'
      '多模态感知智能系统及方法。')
    P(doc,
      '本方案的典型应用场景为人形机器人替代人工完成汽车底盘螺栓拧紧、'
      '发动机螺柱安装、电池包紧固件连接等工序。'
      '此外还可扩展至航空航天紧固件装配、风电设备螺栓维护、'
      '以及其他需要拧紧力矩精确控制的工业场景。')

    P(doc, '2、领域发展状况', bold=True, indent=False)
    P(doc,
      '随着人形机器人进工厂的趋势加速，使用人形机器人替代人工执行螺栓/螺柱装配成为汽车行业的迫切需求。'
      '与传统工业机器人（固定工位、专用工装）不同，人形机器人以类人的双臂+灵巧手配置工作，'
      '在灵活性上接近人工，但在复杂接触任务中面临更大的控制挑战——'
      '因为人形机器人的关节刚度、定位精度不如专用拧紧设备，'
      '遇到螺纹对准偏差、材料摩擦变化等异常时更容易发生滑牙或拧偏。')
    P(doc,
      '当前工业螺栓装配系统的智能化程度很低：'
      '正常装配阶段，绝大多数系统仅依赖预设的速度-力矩曲线执行拧入，'
      '不能根据触觉反馈实时调整策略；'
      '异常发生后，系统仅能通过力矩阈值检测异常，'
      '然后停止等待远程操作员介入（平均响应5-8分钟），产能损失严重。'
      '因此，构建一套覆盖"正常执行+异常处理"的完整智能装配系统，'
      '是人形机器人从试验品走向实际产线部署的关键瓶颈。')
    P(doc,
      '近年来相关技术进展：'
      '大语言模型和多模态基础模型方面——'
      'Pi-Zero(Physical Intelligence 2024)证明VLA模型可以驱动灵巧手操作；'
      'OpenVLA(Stanford 2024)、GR-2(ByteDance 2024)展示了视觉-语言-动作的统一模型；'
      'FuSe(Stanford 2024)证明语言可以作为多模态传感器融合的中间表征。'
      '触觉传感方面——GelSight、BioTac等触觉传感器已被广泛用于接触状态感知。'
      '扩散策略方面——Diffusion Policy(RSS 2023)和Reactive Diffusion Policy(2025)'
      '展示了基于扩散模型的动作生成与触觉反馈融合的可行性。'
      '然而，这些技术尚未被整合为覆盖完整装配生命周期的智能系统。')

    # ============ 1.2 现有技术 ============
    H(doc, '1.2 现有技术方案情况')
    P(doc, '存在与本方案相关但有明显不足的技术方案：', indent=False)

    P(doc,
      '方案A：传统工业拧紧系统（如Atlas Copco、Bosch Rexroth等）。'
      '正常装配阶段执行预设速度-力矩曲线，异常检测依赖阈值规则。'
      '缺点：(1) 正常装配阶段无自适应能力——不能根据触觉/视觉反馈实时调整策略；'
      '(2) 异常只能报警，无法自主诊断原因或执行修正；'
      '(3) 阈值规则需要针对每种产品规格逐一配置。')

    P(doc,
      '方案B：基于深度学习的异常分类（BMW/Fraunhofer 2024等）。'
      '用CNN或LSTM对力矩曲线进行多类别分类。'
      '缺点：(1) 需要大量标注数据（工业异常是小概率事件）；'
      '(2) 只给分类标签，不解释原因也不建议修正方案；'
      '(3) 无法处理训练集中没见过的新异常模式。')

    P(doc,
      '方案C：VLA机器人基础模型（Pi-Zero 2024、OpenVLA 2024等）。'
      '根据视觉+语言指令直接输出动作。'
      '缺点：(1) 没有触觉/力矩输入通道——很多装配异常是视觉看不出来的；'
      '(2) 面向通用任务设计，不适配工业装配精度和安全要求；'
      '(3) 是"执行指令"的模型，不具备诊断和修正能力。')

    P(doc,
      '方案D：视触觉世界模型（OmniVTA 2026、Visuo-Tactile World Models 2026等）。'
      '构建世界模型预测未来接触状态。'
      '缺点：(1) 在线展开计算量大（K条候选×多步展开），实时性差；'
      '(2) 通常与MPC框架结合，与扩散策略等生成式方法不兼容；'
      '(3) 只关注正常操作的状态预测，没有覆盖异常诊断和修正。')

    P(doc, '3、本方案解决的缺点', bold=True, indent=False)
    P(doc,
      '(1) 解决了"正常装配阶段无自适应能力"问题——'
      '多模态感知融合模块实时编码视觉、触觉、力矩信号，'
      '扩散策略根据编码后的感知状态自适应生成拧入动作，'
      '实现了"看-感-做"的闭环控制；')
    P(doc,
      '(2) 解决了"异常只能报警不能诊断"问题——'
      '通过力-触觉信号分词器将连续传感信号转化为LLM可理解的token，'
      '利用大模型的因果推理能力推断异常根因并以自然语言表述；')
    P(doc,
      '(3) 解决了"出了问题只能停机等人"问题——'
      '诊断结论驱动修正动作规划，形成从发现到修正的完整闭环；')
    P(doc,
      '(4) 解决了"系统不具备经验积累能力"问题——'
      '每次装配（正常或异常）的多模态传感数据和诊断结果自动存入案例库，'
      '通过RAG检索和定期微调实现系统持续自进化。')

    P(doc, '参考文献', bold=True, indent=False)
    P(doc,
      '[1] Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023. '
      '[2] Physical Intelligence, "pi0: VLA Flow Model for General Robot Control", 2024. '
      '[3] Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", CoRL 2024. '
      '[4] Xue et al., "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning", 2025. '
      '[5] Zheng et al., "OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Manipulation", 2026. '
      '[6] Ruan et al., "ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer", 2026. '
      '[7] Heng et al., "ViTacFormer: Learning Cross-Modal Representation", 2025. '
      '[8] Stanford, "FuSe: Fusing Sensory Data through Language", CoRL 2024. '
      '[9] Meta FAIR, "Sparsh: Self-supervised Touch Representations", 2024. '
      '[10] Du and Song, "DynaGuide: Steering Diffusion Policies with Dynamic Guidance", 2025.')

    # ============ 1.3 ============
    H(doc, '1.3 在先专利')
    P(doc, '经检索，现有专利集中于：(1) 基于阈值/窗口的异常检测方法；'
      '(2) 基于统计过程控制(SPC)的质量判定；(3) 基于专家规则库的故障树诊断。'
      '未发现将多模态大模型的语义推理能力与扩散策略的自适应动作生成相结合、'
      '覆盖完整装配生命周期（正常执行+异常处理）的在先专利。')

    # ============ 2.1 ============
    H(doc, '2.1 应用场景')
    add_figure(doc, os.path.join(FIG_DIR, 'patent2_fig1_humanoid_scenario.png'),
               '【图1：系统应用场景示意图——人形机器人在汽车车间执行螺栓拧紧作业】')

    P(doc,
      '本方案的典型应用场景为人形机器人在汽车总装车间执行底盘螺栓/螺柱的插入和拧紧作业。'
      '与传统固定式拧紧设备不同，人形机器人以类人的双臂+灵巧手配置工作：'
      '一只手抓取螺栓、将其对准螺纹孔、插入旋拧直至达到目标力矩。'
      '由于人形机器人的关节柔度较大、末端定位精度不如专用设备，'
      '螺栓装配过程中更容易出现对准偏差和拧紧异常。')

    P(doc,
      '螺栓/螺柱装配中的常见异常模式及其力学表征：')
    P(doc,
      '(a) 交叉拧入（cross-threading）——螺栓进入角度倾斜，螺纹未正确啮合而强行旋入。'
      '力曲线表现为：旋转初期扭矩异常偏高（正常自由旋入阶段扭矩接近0，交叉拧入时>0.5N·m），'
      '且随旋转角度线性增大。触觉表现为螺栓头部单侧受力偏大。'
      '如果不及时退出，螺纹将永久损坏。')
    P(doc,
      '(b) 螺栓倾斜未对准——螺栓轴线与螺纹孔轴线有角度偏差（>2°），'
      '螺栓尖端只有部分螺纹扣进入。力曲线表现为旋转阻力波动'
      '（每转一圈出现一次力矩峰值，因为倾斜导致螺纹周期性卡顿）。'
      '触觉表现为灵巧手指尖的接触力呈周期性单侧偏移。')
    P(doc,
      '(c) 螺纹孔内异物/碎屑——孔内有金属碎屑或残留涂料阻塞。'
      '力曲线表现为突发性随机尖峰（碎屑被螺纹挤压时产生），'
      '与固定周期的倾斜波动不同，尖峰出现位置随机。')
    P(doc,
      '(d) 螺栓规格不匹配——错拿了相邻规格的螺栓（如M8误拿为M10），'
      '螺栓完全无法旋入。力曲线表现为首圈旋转时即遇到极大阻力（>5N·m）。')
    P(doc,
      '(e) 力矩不达标/过冲——拧紧过程顺利但最终力矩未达到目标值'
      '或超过目标值上限（材料屈服）。')
    P(doc,
      '每种异常对应不同的修正策略：'
      '交叉拧入→立即反转退出+重新对准角度+轻力矩试探性旋入验证；'
      '倾斜未对准→退出+灵巧手调整夹持姿态使螺栓轴线对正+重新插入；'
      '孔内异物→完全取出螺栓+吹气清洁螺纹孔+重试；'
      '规格不匹配→完全放弃本次+报告异常等待换料；'
      '力矩过冲→松退90°+以更低速重新拧紧。')

    P(doc, '系统硬件配置：', bold=True, indent=False)
    P(doc,
      '(1) 一台人形机器人（如小米CyberOne、特斯拉Optimus等双臂构型），'
      '具备双臂协作能力，每臂6-7自由度，灵巧手具有力感知的手指；')
    P(doc,
      '(2) 灵巧手指尖集成触觉传感单元（压力阵列或GelSight薄膜传感器），'
      '能够检测抓取螺栓时的接触力分布和螺栓姿态偏移，采样率30Hz；')
    P(doc,
      '(3) 腕部六维力/力矩传感器（500Hz采样），'
      '检测拧紧过程中的轴向力Fz、径向力Fx/Fy和扭矩Tz——'
      '其中Tz是判断拧紧质量的核心信号；')
    P(doc,
      '(4) 人形机器人头部双目相机（观察螺栓和孔位的对准状态）'
      '+ 腕部近距相机（近距离观察螺栓尖端与螺纹孔口的对准精度）；')
    P(doc,
      '(5) 机器人内置边缘计算模块（搭载NVIDIA Jetson AGX Orin或同级嵌入式GPU），'
      '运行感知编码、动作生成、异常检测、大模型推理等算法；')
    P(doc,
      '(6) 远程监控终端（操作员可通过屏幕查看装配状态和诊断报告，远程确认/否决修正计划）。')

    # ============ 2.2 ============
    H(doc, '2.2 技术问题')
    P(doc, '本方案解决以下技术问题：')
    P(doc,
      '第一，如何在正常装配阶段实现多模态感知融合的自适应拧入控制。'
      '传统系统执行预设力矩曲线，不利用触觉和视觉反馈。'
      '本方案需要将视觉（螺栓对准状态）、触觉（接触力分布）、'
      '力矩（拧紧过程力/力矩曲线）和本体感知（关节角度）'
      '四路信号实时融合，为动作生成提供丰富的感知上下文。')
    P(doc,
      '第二，如何在正常装配过程中实时监测异常征兆。'
      '不能等到力矩大幅超标才报警，而要在异常萌芽阶段（如力矩刚开始异常偏高时）'
      '就能检测到。这需要一个能在毫秒级延迟下持续运行的异常监测模块。')
    P(doc,
      '第三，如何从多模态传感信号中判断异常的具体类型和根本原因。'
      '这不是简单的力超限报警，而是要做因果推理——'
      '例如力曲线在某深度陡增，需要结合触觉分布（偏左？均匀？）'
      '和视觉姿态（有偏移？有旋转？）来区分是对准问题还是卡滞还是异物。')
    P(doc,
      '第四，如何让诊断结论以人类可理解的形式呈现，'
      '并根据诊断结论自主规划正确的修正动作。')
    P(doc,
      '第五，如何保证整个系统的安全性并实现持续自进化。'
      '需要根据修正动作的风险等级实行分级管控，'
      '并通过经验积累机制让系统越用越聪明。')


    # ============ 2.3 ============
    H(doc, '2.3 技术方案完整内容')

    add_figure(doc, os.path.join(FIG_DIR, 'patent2_fig2_system_architecture.png'),
               '【图2：系统整体架构图——五层架构：感知层→编码层→决策层→执行层→经验层】')

    P(doc,
      '本方案是一种覆盖完整装配生命周期的智能系统，由五大模块组成：'
      '(1) 多模态感知编码模块——实时编码视觉、触觉、力矩和本体感知四路信号；'
      '(2) 自适应拧入动作生成模块——基于扩散策略的自适应动作生成；'
      '(3) 实时异常监测模块——毫秒级异常征兆检测；'
      '(4) 大模型智能诊断与修正模块——利用LLM因果推理诊断异常并规划修正；'
      '(5) 经验积累与自进化模块——案例库+RAG检索+定期微调。'
      '五者组合形成"感知→生成→监测→诊断→进化"的完整智能装配闭环。',
      indent=False)

    # ==== 阶段一：正常装配 ====
    P(doc, '阶段一：多模态感知融合的自适应拧入执行', bold=True, indent=False)

    P(doc, '1.1 多模态感知编码', bold=True, indent=False)
    P(doc,
      '系统在正常装配过程中持续采集并编码四路信号：')
    P(doc,
      '(a) 视觉编码：全局相机和腕部相机图像分别通过预训练的ResNet18骨干网络提取特征图，'
      '经1×1卷积投影到512维空间，得到视觉token序列。全局相机提供整体场景信息'
      '（螺栓相对于螺纹孔的位置关系），腕部相机提供近距细节'
      '（螺栓尖端与螺纹孔口的对准精度）。')
    P(doc,
      '(b) 触觉编码：触觉传感器输出的标志点位移序列(9×9×2)通过PointNet风格的编码器'
      '编码为512维触觉embedding。标志点位移反映了接触面上的力分布，'
      '能够感知螺栓抓取姿态、接触点偏移等视觉无法获取的信息。')
    P(doc,
      '(c) 力矩编码：腕部六维力/力矩传感器的实时数据(Fx, Fy, Fz, Tx, Ty, Tz)'
      '通过滑动窗口（最近1秒）提取时域特征，经一维卷积编码为64维向量。'
      '其中Tz（绕螺栓轴线的扭矩）是判断拧入质量的核心信号。')
    P(doc,
      '(d) 本体感知编码：机器人当前7维关节角度经归一化后作为7维向量输入。')
    P(doc,
      '四路编码后的特征向量拼接为统一的感知表示（共512+512+64+7=1095维），'
      '送入下游的动作生成和异常监测模块。')

    P(doc, '1.2 基于扩散策略的自适应拧入动作生成', bold=True, indent=False)
    P(doc,
      '正常装配阶段的动作由扩散策略（Diffusion Policy）生成。'
      '扩散策略以当前多模态感知编码为条件，通过DDPM去噪过程生成未来20步的动作序列'
      '（7维关节角度）。与传统预设力矩曲线不同，扩散策略能根据当前感知状态'
      '自适应地生成拧入动作——例如当视觉检测到螺栓略有偏斜时，'
      '策略会自动微调关节角度进行补偿，而无需人工编程。')
    P(doc,
      '扩散策略的训练使用正常装配的专家示范数据（约269个episode），'
      '每个episode包含完整的"抓取→对准→旋拧→力矩达标"序列。'
      '训练时以多模态感知编码为条件，以专家动作为目标，'
      '通过条件DDPM的去噪目标进行端到端训练。')

    # ==== 阶段二：实时异常监测 ====
    P(doc, '阶段二：实时异常监测模块', bold=True, indent=False)
    P(doc,
      '在正常装配执行的同时，系统并行运行一个轻量级异常监测模块，'
      '实时检测拧入过程中的异常征兆。该模块与动作生成模块共享多模态感知编码，'
      '额外增加的计算开销极小（<5ms），不影响控制实时性。')
    P(doc,
      '异常监测模块由一个单层LSTM + 阈值分类头组成：'
      'LSTM接收最近1秒的力矩特征序列（62个时间步×64维），'
      '输出异常概率分数p_anomaly。当p_anomaly超过阈值θ（默认0.7）时，'
      '触发阶段三的智能诊断流程。')
    P(doc,
      '该监测模块的核心优势在于：它不是简单的阈值比较，'
      '而是学习了力矩曲线的正常模式——'
      '即使力矩绝对值没有超标，但如果曲线形状偏离了正常模式'
      '（如正常应在15°处开始缓慢上升，实际在10°处就开始异常偏高），'
      '也能被检测到。')

    # ==== 阶段三：智能诊断 ====
    add_figure(doc, os.path.join(FIG_DIR, 'patent2_fig4_signal_tokenizer.png'),
               '【图3：力-触觉信号分词器网络结构详图】')

    P(doc, '阶段三：大模型智能诊断与分级修正', bold=True, indent=False)

    P(doc, '3.1 力-触觉信号分词器（Force-Tactile Tokenizer）【核心算法创新之一】', bold=True, indent=False)
    P(doc,
      '当异常监测模块触发诊断后，首先将连续传感信号"翻译"为LLM可理解的离散token。'
      '这是本方案的关键创新——将连续物理信号桥接到大语言模型的语义空间。')
    P(doc,
      '分词器包含三个分支：')
    P(doc,
      '(a) 力矩信号分支：输入为最近1秒的六维力/力矩序列(500×6)。'
      '经过一维因果卷积网络（3层，核宽7，通道6→64→128→256，步长2/2/2下采样，'
      '序列长度500→62），再经双向GRU（hidden=256）建模全局时序依赖。'
      '最后通过向量量化层（VQ, codebook=512个码字，维度=4096）'
      '将连续特征量化为离散token，输出62个力矩token。')
    P(doc,
      '(b) 触觉信号分支：输入为最近1秒的触觉序列(30帧×9×9×2)。'
      '采用二维空间卷积+时间因果卷积交替结构，'
      '经独立codebook（256个码字）量化后输出8个触觉token。')
    P(doc,
      '(c) 视觉分支：腕部近距相机图像经ViT-B/14提取CLS token(768维)，'
      '投影到4096维后作为1个视觉token（连续嵌入，不做量化）。')
    P(doc,
      '分词器总输出：62+8+1=71个"传感器token"，与文字token拼接后送入LLM。'
      '训练分两阶段：先用10万条力矩曲线做自监督重建预训练（VQ-VAE风格），'
      '再用500条标注数据做诊断对齐微调（LoRA）。')

    P(doc, '3.2 领域微调多模态大模型（诊断推理核心）', bold=True, indent=False)
    P(doc,
      '采用7B参数量的多模态大语言模型（如InternVL2-7B）作为推理骨干。'
      '模型输入由三部分拼接：'
      '(1) 系统指令token（定义角色、输出格式、安全约束）；'
      '(2) RAG检索到的Top-3历史案例（约200-400个token）；'
      '(3) 当前71个传感器token + 结构化状态描述（关节角度、目标力矩等）。')
    P(doc,
      '模型输出结构化JSON：reasoning（推理过程，自然语言）、'
      'root_cause（根因类别）、confidence（0-1）、direction（修正方向）。')

    P(doc, 'RAG案例检索', bold=True, indent=False)
    P(doc,
      '案例检索使用力矩曲线对比学习训练的专用嵌入模型（非通用text-embedding）。'
      '该模型结构为分词器encoder（冻结）+ 2层MLP投影头(256→128→64)。'
      '训练用监督对比学习（SupCon Loss, τ=0.07），'
      '重点挖掘"表面相似但原因不同"的困难负例（如交叉拧入vs倾斜进入）。'
      '在线检索时，当前力矩曲线编码为64维向量，做余弦相似度Top-3检索。')

    P(doc, 'LLM领域微调', bold=True, indent=False)
    P(doc,
      '采用LoRA（rank=16, alpha=32）对LLM注意力层Q/V矩阵做低秩适配。'
      '训练数据约800条标注异常案例。传感器token分配独立可学习位置嵌入，'
      '力矩/触觉/视觉token之间插入[FORCE]、[TACTILE]、[VISION]分隔token。'
      '损失为下一token预测交叉熵，仅在输出JSON部分计算。')

    P(doc, '3.3 带几何约束的修正参数回归头【核心算法创新之二】', bold=True, indent=False)
    P(doc,
      'LLM对精确数值生成不够可靠，因此在LLM最后一层隐状态上接一个参数回归头，'
      '直接从隐层表示回归精确修正参数。')
    P(doc,
      '输入：LLM最后一层[EOS]位置的隐状态h∈R^4096。'
      '网络：3层MLP（4096→512→256→8）。')
    P(doc,
      '8维修正参数向量定义：\n'
      '  p[0]: 退出圈数 (0-3圈)\n'
      '  p[1]: 是否完全拔出 (0/1)\n'
      '  p[2]: 角度补偿-俯仰(°) (±5°)\n'
      '  p[3]: 角度补偿-偏航(°) (±5°)\n'
      '  p[4]: 位置补偿-X(mm) (±2mm)\n'
      '  p[5]: 位置补偿-Y(mm) (±2mm)\n'
      '  p[6]: 重新拧入最大力矩(N·m)\n'
      '  p[7]: 重新拧入转速(rpm)')
    P(doc,
      '几何约束层：\n'
      '  - 各参数通过tanh映射到物理允许范围；\n'
      '  - 位置补偿通过视觉测量做soft gating：\n'
      '    p_final[4:6] = σ(gate) × p_regression[4:6] + (1-σ(gate)) × p_visual[4:6]\n'
      '  - 当root_cause="规格不匹配"时，所有修正参数置零（正确操作是放弃本次）。')
    P(doc,
      '回归头单独训练（不与LLM联合训练），'
      '从人工修正记录中提取GT标签（约800条）。'
      '损失 = SmoothL1主损失 + 物理范围软约束 + 修正类型分类辅助损失。')

    P(doc, '3.4 安全分级管控机制', bold=True, indent=False)
    P(doc,
      '根据confidence值和修正参数幅度进行三级安全分级：')
    P(doc,
      '低风险（直接执行）：仅退出动作（p[0]>0且p[2:6]全为0），无需人工确认。')
    P(doc,
      '中风险（默认确认）：角度补偿<3°且位置补偿<0.5mm，'
      '3秒内操作员未否决则自动执行。')
    P(doc,
      '高风险（必须确认）：补偿量大、confidence<0.6、或root_cause=规格不匹配，'
      '必须等待操作员明确确认。')

    # ==== 阶段四：经验积累 ====
    P(doc, '阶段四：经验积累与持续自进化', bold=True, indent=False)
    P(doc,
      '每次装配操作（无论正常完成还是异常处理）的多模态传感数据和诊断结果'
      '自动存入结构化案例库。案例库采用FAISS索引加速检索。')
    P(doc,
      '系统通过三种机制实现持续自进化：')
    P(doc,
      '(1) 在线学习——每次成功的修正记录实时追加到案例库，'
      'RAG检索的候选集不断扩大，新异常模式可被快速覆盖；')
    P(doc,
      '(2) 定期微调——积累到一定量新数据后（如每月），'
      '对LLM和分词器做增量LoRA微调，提升诊断准确率；')
    P(doc,
      '(3) 异常监测模块重训练——随着正常装配数据的积累，'
      '定期用最新的正常模式数据重训练LSTM异常检测器，'
      '提高检测灵敏度、降低误报率。')
    P(doc,
      '运行半年后，案例库覆盖度可达90%以上的常见异常模式，'
      '诊断正确率从初始的约70%逐步提升至90%+。')

    # ==== 在线完整流程 ====
    add_figure(doc, os.path.join(FIG_DIR, 'patent2_fig3_diagnosis_workflow.png'),
               '【图4：完整智能装配闭环工作流程图】')

    P(doc, '在线完整流程', bold=True, indent=False)
    P(doc,
      'Step 1（多模态感知编码，持续运行）：'
      '视觉、触觉、力矩、本体感知四路信号实时编码为统一感知表示。')
    P(doc,
      'Step 2（自适应动作生成，持续运行）：'
      '扩散策略以感知表示为条件，生成未来20步自适应拧入动作。')
    P(doc,
      'Step 3（实时异常监测，并行运行）：'
      'LSTM异常检测器持续分析力矩曲线模式，输出异常概率。')
    P(doc,
      'Step 4（触发智能诊断，异常时触发，~1.5-2s）：'
      '信号分词器编码→RAG案例检索→LLM推理→修正参数回归。')
    P(doc,
      'Step 5（安全分级+修正执行，~2-15s）：'
      '根据风险等级执行修正动作，含力保护和重试机制。')
    P(doc,
      'Step 6（验证+经验积累）：'
      '修正后重新拧入验证，完整记录存入案例库。')

    # ---- 关键步骤总结 ----
    P(doc, '关键步骤总结', bold=True, indent=False)
    P(doc,
      '(1) 多模态感知融合模块：将视觉、触觉、力矩、本体感知四路信号统一编码，'
      '为下游动作生成和异常诊断提供丰富的感知上下文。')
    P(doc,
      '(2) 自适应拧入动作生成：基于扩散策略根据实时感知状态生成拧入动作，'
      '实现"看-感-做"闭环，替代传统预设力矩曲线。')
    P(doc,
      '(3) 轻量级实时异常监测：LSTM持续分析力矩模式，毫秒级检测异常征兆，'
      '与动作生成并行运行不增加控制延迟。')
    P(doc,
      '(4) 力-触觉信号分词器：通过因果卷积+向量量化将连续传感信号转化为离散token，'
      '打通传感器信号到大语言模型的语义鸿沟。')
    P(doc,
      '(5) LLM因果推理诊断：大模型的思维链推理提供可解释的诊断过程，'
      '结合RAG案例检索实现对新异常模式的零样本适应。')
    P(doc,
      '(6) 带几何约束的修正参数回归：从LLM隐状态直接回归精确修正参数，'
      '视觉-回归soft gating融合两种精度来源。')
    P(doc,
      '(7) 三级安全分级管控：根据置信度和风险等级灵活决策——该果断时果断、该谨慎时谨慎。')
    P(doc,
      '(8) 经验积累自进化：案例库+定期微调实现系统越用越聪明，'
      '诊断正确率从初始70%逐步提升至90%+。')

    # ============ 2.4 ============
    H(doc, '2.4 有益效果')
    P(doc,
      '(1) 正常装配阶段实现自适应智能控制。'
      '与传统预设力矩曲线相比，扩散策略根据实时多模态感知自适应生成拧入动作，'
      '能够在螺栓略有偏斜时自动微调补偿，减少正常装配过程中的异常发生率。')
    P(doc,
      '(2) 大幅减少异常停机导致的产能损失。'
      '传统系统异常后等待人工处理平均3-5分钟/次，'
      '本方案对可自主修正的异常处理时间约8-20秒，'
      '产能恢复速度提升约10-20倍。')
    P(doc,
      '(3) 诊断结论透明可解释。'
      '整个推理过程以自然语言输出（包括"为什么排除了其他可能性"），'
      '质量工程师可以直接阅读审查，便于工艺问题的根因追溯。')
    P(doc,
      '(4) 对新异常类型具有零样本适应能力。'
      '大模型结合RAG历史案例做类比推断，'
      '即使遇到未见过的异常模式也能给出合理诊断。')
    P(doc,
      '(5) 系统持续自进化。'
      '每次操作都在积累经验，运行半年后诊断正确率可从70%提升至90%+。')
    P(doc,
      '(6) 完整覆盖装配生命周期。'
      '从正常执行到异常处理的完整闭环，'
      '避免了"正常阶段用一套系统、异常阶段用另一套"的割裂问题，'
      '系统架构统一、数据共享、模块解耦可独立升级。')
    P(doc,
      '(7) 安全分级机制兼顾自主性与安全性。'
      '不是冒险的全自动方案，也不是保守的全人工方案，'
      '而是根据风险等级灵活决策的分级管控方案。')

    # ============ 附件 ============
    H(doc, '附件')
    P(doc, '参考文献：', bold=True, indent=False)
    P(doc,
      '[1] Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023. '
      '[2] Physical Intelligence, "pi0: VLA Flow Model for General Robot Control", 2024. '
      '[3] Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", CoRL 2024. '
      '[4] Xue et al., "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning", 2025. '
      '[5] Zheng et al., "OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Manipulation", 2026. '
      '[6] Ruan et al., "ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer", 2026. '
      '[7] Heng et al., "ViTacFormer: Learning Cross-Modal Representation", 2025. '
      '[8] Stanford, "FuSe: Fusing Sensory Data through Language", CoRL 2024. '
      '[9] Meta FAIR, "Sparsh: Self-supervised Touch Representations", 2024. '
      '[10] Du and Song, "DynaGuide: Steering Diffusion Policies with Dynamic Guidance", 2025. '
      '[11] BMW/Fraunhofer, "DL-Based Quality Prediction for Robotic Fastener Assembly", RCIM 2024. '
      '[12] Toyota Research, "F/T Guided Bolt Tightening via Deep RL", RA-L 2024.')

    out = '/home/chenshuai/Project/TactileACT-cs/paper/patent/patent2_complete_v3.docx'
    doc.save(out)
    print(f'Patent saved to: {out}')


if __name__ == '__main__':
    build()
