#!/usr/bin/env python3
"""
专利交底书（最终版，含图片嵌入）：
一种基于多模态感知与大模型推理的人形机器人螺栓装配异常自主诊断及分级修正方法与系统

输出: patent2_final_with_figures.docx
"""

import os
from docx import Document
from docx.shared import Pt as DocPt, Cm as DocCm, Inches as DocIn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def set_cn_font(run, name='宋体', size=DocPt(12)):
    """设置中文字体，兼容 eastAsia 字体映射。"""
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


def set_heading_font(paragraph, text, name='黑体', size=DocPt(14)):
    """对 heading 段落设置中文字体（黑体）。"""
    for run in paragraph.runs:
        set_cn_font(run, name, size)


def P(doc, text, bold=False, indent=True):
    """正文段落：宋体 12pt，首行缩进 0.74cm。"""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = DocCm(0.74)
    run = p.add_run(text)
    run.bold = bold
    set_cn_font(run, '宋体', DocPt(12))
    return p


def H(doc, text, level=2):
    """标题段落：黑体 14pt。"""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        set_cn_font(run, '黑体', DocPt(14))
    return heading


def add_figure(doc, filename, caption, width_cm=14):
    """嵌入图片并添加居中图注。"""
    fig_path = os.path.join(FIGURE_DIR, filename)
    if not os.path.isfile(fig_path):
        print(f'  [WARNING] 图片不存在，跳过: {fig_path}')
        P(doc, caption, bold=True, indent=False)
        return

    # 图片段落（居中）
    p_fig = doc.add_paragraph()
    p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_fig = p_fig.add_run()
    run_fig.add_picture(fig_path, width=DocCm(width_cm))

    # 图注段落（居中、加粗、宋体 10.5pt）
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_cap = p_cap.add_run(caption)
    run_cap.bold = True
    set_cn_font(run_cap, '宋体', DocPt(10.5))


def build():
    doc = Document()

    # ---- 页面设置 ----
    for s in doc.sections:
        s.top_margin = DocCm(2.54)
        s.bottom_margin = DocCm(2.54)
        s.left_margin = DocCm(3.17)
        s.right_margin = DocCm(3.17)

    # ---- 封面标题 ----
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('技术交底书')
    r.bold = True
    set_cn_font(r, '黑体', DocPt(18))
    doc.add_paragraph()

    P(doc, '发明名称：一种基于多模态感知与大模型推理的人形机器人螺栓装配异常自主诊断及分级修正方法与系统', bold=True, indent=False)
    doc.add_paragraph()

    # ============ 验证性 ============
    H(doc, '验证性')
    P(doc, '容易验证。', bold=True, indent=False)
    P(doc,
        '取证方式：(1) 人为制造一个螺栓装配异常（如将螺栓故意倾斜3°放入螺纹孔），观察系统是否能自动检出并生成诊断结论；'
        '(2) 检查系统是否输出了自然语言形式的异常描述和修正计划文本（可从系统日志或人机界面截图取证）；'
        '(3) 观察系统是否在诊断后自主执行了修正动作（如退出螺栓→调整姿态→重新对准拧入），而非仅停机报警；'
        '(4) 检查系统是否包含"安全分级"逻辑——低风险修正直接执行、中风险需操作员默认确认、高风险强制转人工。')

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
        '面向人形机器人在汽车产线执行螺栓/螺柱插拧任务时，能够自主感知拧紧异常状态、'
        '利用多模态大模型推理诊断异常原因、'
        '并根据安全分级策略自主规划和执行修正动作的控制系统与方法。')
    P(doc,
        '本方案的典型应用场景为人形机器人替代人工完成汽车底盘螺栓拧紧、'
        '发动机螺柱安装、电池包紧固件连接等工序。'
        '此外还可扩展至航空航天紧固件装配、风电设备螺栓维护、'
        '以及其他需要拧紧力矩精确控制且允许自主修正的工业场景。')

    P(doc, '2、领域发展状况', bold=True, indent=False)
    P(doc,
        '随着人形机器人进工厂的趋势加速，使用人形机器人替代人工执行螺栓/螺柱装配成为汽车行业的迫切需求。'
        '与传统工业机器人（固定工位、专用工装）不同，人形机器人以类人的双臂+灵巧手配置工作，'
        '在灵活性上接近人工，但在复杂接触任务中面临更大的控制挑战——'
        '因为人形机器人的关节刚度、定位精度不如专用拧紧设备，'
        '遇到螺纹对准偏差、材料摩擦变化等异常时更容易发生滑牙或拧偏。')
    P(doc,
        '当前人形机器人在螺栓装配中遇到异常时的处理方式非常原始：'
        '绝大多数系统仅能通过力矩阈值检测异常（如扭矩突增超过设定值），'
        '然后停止等待远程操作员介入。'
        '由于人形机器人部署位置分散（可能在车间不同工位流动作业），'
        '操作员响应时间更长（平均5-8分钟），产能损失更为严重。'
        '因此，赋予人形机器人"自主诊断+自主修正"的能力，'
        '是其从试验品走向实际产线部署的关键瓶颈。')
    P(doc,
        '近年来相关技术进展：'
        '大语言模型和多模态基础模型方面——'
        'Pi-Zero(Physical Intelligence 2024)证明VLA模型可以驱动灵巧手操作；'
        'OpenVLA(Stanford 2024)、GR-2(ByteDance 2024)展示了视觉-语言-动作的统一模型；'
        'FuSe(Stanford 2024)证明语言可以作为多模态传感器融合的中间表征。'
        '触觉传感方面——安装在灵巧手指尖的触觉传感器（如GelSight、BioTac）'
        '能够感知螺栓与螺纹孔之间的接触状态；'
        '六维力/力矩传感器能实时监测拧紧力矩和径向偏载。'
        '然而，这些技术尚未被整合为人形机器人螺栓装配场景的异常诊断和自主修正系统。')

    # ============ 1.2 现有技术 ============
    H(doc, '1.2 现有技术方案情况')
    P(doc, '存在与本方案相关但有明显不足的技术方案：', indent=False)

    P(doc,
        '方案A：传统工业异常检测系统（如Atlas Copco ToolsNet、Bosch Rexroth拧紧监控等）。'
        '通过预设的力矩阈值、角度窗口、梯度限制等规则判断装配过程是否异常。'
        '缺点：(1) 只能判断"有没有异常"，无法判断"什么异常"和"为什么异常"——'
        '比如力矩突增可能是对准偏差、可能是异物、也可能是密封圈卡滞，'
        '系统无法区分，只能笼统报警；'
        '(2) 检测到异常后唯一的处理方式是停机等人工，没有任何自主修正能力；'
        '(3) 阈值规则需要工程师针对每种产品规格逐一配置和调优，维护成本高。')
    P(doc,
        '方案B：基于深度学习的异常分类（BMW/Fraunhofer 2024、多家车企内部方案）。'
        '用CNN或LSTM对力矩曲线进行多类别分类（正常/滑牙/贴合不良/力矩不足等）。'
        '相比方案A能给出异常类型。'
        '缺点：(1) 需要对每种异常类型采集大量标注数据——而工业异常是小概率事件，'
        '有些异常类型一个月才出现几次，标注数据获取困难；'
        '(2) 只给出分类标签，不解释原因（如"滑牙"但不说为什么滑牙），也不建议修正方案；'
        '(3) 无法处理训练集中没见过的新异常模式（输出"未知"或错误分类）；'
        '(4) 仍然没有修正能力。')
    P(doc,
        '方案C：VLA机器人基础模型（Pi-Zero 2024、OpenVLA 2024、GR-2 2024等）。'
        '这些模型根据视觉+语言指令直接输出机器人动作，具有一定泛化能力。'
        '缺点：(1) 没有触觉/力矩输入通道——在装配任务中，'
        '很多异常是视觉看不出来但力矩能感知到的（如螺纹内部的交叉拧入）；'
        '(2) 面向通用家庭/实验室任务设计，不适配工业装配的精度（亚毫米级）和安全要求；'
        '(3) 是"执行指令"的模型而非"诊断问题"的模型——'
        '它们被设计为"告诉我做什么我就去做"，而不是"自己发现问题然后决定怎么修"。')
    P(doc,
        '方案D：基于强化学习的自适应装配策略（Toyota Research 2024等）。'
        '用RL学习当力矩反馈异常时调整动作策略。'
        '缺点：(1) 策略是黑盒的——操作员看不懂为什么机器人做了某个修正动作，'
        '缺乏可解释性，在安全敏感的工业场景中难以通过审核；'
        '(2) RL策略依赖训练分布，在未见过的新异常模式下可能输出危险动作；'
        '(3) 没有诊断环节——不区分异常原因就执行修正，修正方式可能对不上真实原因。')

    P(doc, '3、本方案解决的缺点', bold=True, indent=False)
    P(doc,
        '(1) 解决了"只知其然不知其所以然"问题——'
        '通过多模态大模型的因果推理能力，不仅检出异常，'
        '还推断出具体原因并以自然语言表述'
        '（如"力矩在30°处陡增，根据触觉偏左和视觉偏移综合判断，是进入角度偏斜所致"）；'
        '(2) 解决了"只能停机等人"问题——'
        '诊断结论驱动修正动作规划，形成从发现到修正的完整闭环；'
        '(3) 解决了"对新异常无能为力"问题——'
        '大模型具有零样本推理能力，结合RAG调取历史案例，即使未见过的异常模式也能给出合理诊断；'
        '(4) 解决了"不可解释"问题——'
        '诊断过程和修正计划全部以自然语言输出，可供质量工程师审查和经验积累。')

    P(doc, '5、参考文献', bold=True, indent=False)
    P(doc,
        '[1] Physical Intelligence, "pi0: VLA Flow Model for General Robot Control", 2024. '
        '[2] Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", CoRL 2024. '
        '[3] Stanford, "FuSe: Fusing Sensory Data for Robot Learning through Language", CoRL 2024. '
        '[4] Meta FAIR, "Sparsh: Self-supervised Touch Representations", 2024. '
        '[5] BMW/Fraunhofer, "DL-Based Quality Prediction for Robotic Fastener Assembly", RCIM 2024. '
        '[6] ByteDance, "GR-2: Generative Video-Language-Action Model", 2024. '
        '[7] Toyota Research, "F/T Guided Bolt Tightening via Deep RL", RA-L 2024. '
        '[8] Zheng et al., "OmniVTA: Visuo-Tactile World Modeling", 2026. '
        '[9] Xue et al., "Reactive Diffusion Policy", 2025. '
        '[10] Ruan et al., "ReTac-ACT: State-Gated Vision-Tactile Fusion", 2026.')

    # ============ 1.3 ============
    H(doc, '1.3 在先专利')
    P(doc, '经检索，现有专利集中于：(1) 基于阈值/窗口的异常检测方法；'
        '(2) 基于统计过程控制(SPC)的质量判定；(3) 基于专家规则库的故障树诊断。'
        '未发现将多模态大模型的语义推理能力应用于实时装配异常根因诊断、'
        '并据此自主规划修正动作的在先专利。')

    # ============ 2.1 应用场景 ============
    H(doc, '2.1 应用场景')

    # ---- 图1：应用场景 ----
    add_figure(doc, 'patent2_fig1_humanoid_scenario.png',
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
        '与固定周期的倾斜波动不同，尖峰出现位置随机。'
        '触觉表现为不规则突变。')
    P(doc,
        '(d) 螺栓规格不匹配——错拿了相邻规格的螺栓（如M8误拿为M10），'
        '螺栓完全无法旋入。力曲线表现为首圈旋转时即遇到极大阻力（>5N·m），'
        '远超正常自由旋入阶段的值。视觉可观察到螺栓明显无法进入孔位。')
    P(doc,
        '(e) 力矩不达标/过冲——拧紧过程顺利但最终力矩未达到目标值'
        '或超过目标值上限（材料屈服）。这不是过程中的异常，而是结果异常，'
        '通常需要松退后重新拧紧。')
    P(doc,
        '每种异常对应不同的修正策略：'
        '交叉拧入→立即反转退出+重新对准角度+轻力矩试探性旋入验证；'
        '倾斜未对准→退出+灵巧手调整夹持姿态使螺栓轴线对正+重新插入；'
        '孔内异物→完全取出螺栓+吹气清洁螺纹孔+重试；'
        '规格不匹配→完全放弃本次+报告异常等待换料；'
        '力矩过冲→松退90°+以更低速重新拧紧。'
        '如果搞反修正方式（如对交叉拧入施加更大扭矩强行通过），将直接损坏螺纹，零件报废。')

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
        '(4) 人形机器人头部双目相机（模拟人眼视野，观察螺栓和孔位的对准状态）'
        '+ 腕部近距相机（近距离观察螺栓尖端与螺纹孔口的对准精度）；')
    P(doc,
        '(5) 机器人内置边缘计算模块（搭载NVIDIA Jetson AGX Orin或同级嵌入式GPU），'
        '运行异常检测、大模型推理、视觉测量等算法；')
    P(doc,
        '(6) 远程监控终端（操作员可通过屏幕查看诊断结论，远程确认/否决修正计划）。')

    # ============ 2.2 技术问题 ============
    H(doc, '2.2 技术问题')
    P(doc, '本方案解决以下技术问题：')
    P(doc,
        '第一，如何从多模态传感信号中判断异常的具体类型和根本原因。'
        '这不是简单的力超限报警，而是要做因果推理——'
        '例如力曲线在某深度陡增，需要结合触觉分布（偏左？均匀？）'
        '和视觉姿态（有偏移？有旋转？）来区分是对准问题还是卡滞还是异物。'
        '不同传感器看到的是同一物理现象的不同侧面，需要综合分析才能得出正确结论。')
    P(doc,
        '第二，如何让诊断结论以人类可理解的形式呈现。'
        '工业场景要求可追溯和可审查，'
        '黑盒输出（如异常代码0x3A）对工艺改进毫无帮助，'
        '而自然语言描述（如"进入角偏左约2°导致右侧摩擦增大"）'
        '可以直接指导工装调整和来料标准优化。')
    P(doc,
        '第三，如何根据诊断结论自主规划正确的修正动作。'
        '不同异常原因对应不同修正策略，搞反了会加重损坏。'
        '需要系统能从诊断结论推导出对应的修正方案，'
        '并通过定量视觉测量精确化修正参数（如"偏左多少mm"）。')
    P(doc,
        '第四，如何保证修正过程的安全性。'
        '修正动作本身如果不受控也可能损坏零件。'
        '需要根据修正动作的风险等级实行分级管控：'
        '低风险可自主执行、高风险需人工确认。')

    # ============ 2.3 技术方案完整内容 ============
    H(doc, '2.3 技术方案完整内容')

    # ---- 图2：系统架构 ----
    add_figure(doc, 'patent2_fig2_system_architecture.png',
               '【图2：算法整体架构图——信号分词器→LLM推理→修正参数回归头】')

    P(doc,
        '本方案是一种方法流程类方案，核心包含三个自研算法模块：'
        '(1) 力-触觉信号分词器（Force-Tactile Tokenizer），将连续传感信号转化为大模型可处理的离散token；'
        '(2) 领域微调的多模态大模型，接收信号token做思维链推理诊断；'
        '(3) 带几何约束的修正参数回归头（Correction Head），'
        '从大模型隐层状态直接回归出精确的修正动作参数。'
        '三者组合形成"编码→推理→回归"的完整算法链路。', indent=False)

    # ---- 概述 ----
    P(doc,
        '输入：拧紧过程中的六维力/力矩时序信号（最近1秒，500×6维）、'
        '触觉传感器标志点位移（最近1秒，30×9×9×2维）、'
        '腕部近距相机图像（1帧）、当前关节角度（7维）。')
    P(doc,
        '输出：(1) 异常类型诊断及自然语言解释；'
        '(2) 修正动作参数向量（包含退出圈数、补偿角度、补偿位移等精确数值）；'
        '(3) 诊断置信度（用于安全分级决策）。')

    # ==== 模块一：Force-Tactile Tokenizer（核心创新） ====
    P(doc, '模块一：力-触觉信号分词器（Force-Tactile Tokenizer）【核心算法创新之一】', bold=True, indent=False)
    P(doc,
        '该模块的目标是：把连续的力矩时序信号和触觉空间信号"翻译"为大语言模型词表空间中的'
        '离散token序列，使得LLM能像理解文字一样理解传感器信号。'
        '这不是简单的模板文字描述，而是通过训练得到的"传感器词汇表"。')
    P(doc,
        '为什么需要信号分词器而非直接文字描述：'
        '(1) 模板描述只能提取预设特征，遇到模板没覆盖的异常模式就会遗漏；'
        '(2) 连续信号中蕴含的细微模式（如力矩微小的周期性波动只有0.1N·m）'
        '用文字很难精确表达，但训练好的编码器可以捕捉到；'
        '(3) 端到端训练可以让分词器自动学到"对诊断最有帮助的特征"，而非人工设计特征。')

    P(doc, '分词器网络结构：', bold=True, indent=False)
    P(doc,
        '力矩信号分支：输入为最近1秒的六维力/力矩序列(500×6)。'
        '首先经过一维因果卷积网络（3层，核宽7，通道6→64→128→256，步长2/2/2下采样，'
        '序列长度500→62），提取局部时域特征。'
        '然后经过一层双向GRU（hidden=256）建模全局时序依赖，'
        '输出特征序列(62×256)。'
        '最后通过可学习的向量量化层（VQ, codebook大小=512个码字，每个码字维度=LLM隐层维度D=4096）'
        '将连续特征量化为离散token：'
        '每个时间步的256维特征通过线性投影到D=4096维后，'
        '寻找codebook中最近邻的码字，得到该步的token ID。'
        '最终力矩分支输出62个离散token。')
    P(doc,
        '触觉信号分支：输入为最近1秒的触觉序列(30帧×9×9×2)。'
        '采用二维空间卷积(3×3核)+时间维度因果1D卷积的交替结构：'
        '空间卷积(2→32通道, 9×9→5×5) → 时间因果卷积(30→15帧) → '
        '空间卷积(32→64, 5×5→3×3) → 时间因果卷积(15→8帧) → '
        '空间全局池化(3×3→1×1) → 得到(8×64)特征。'
        '同样经过向量量化（共享codebook或独立codebook均可，本方案使用独立codebook=256个码字），'
        '输出8个离散token。')
    P(doc,
        '视觉分支：腕部近距相机图像经过预训练的ViT-B/14提取CLS token(768维)，'
        '经线性投影到D=4096维后作为1个视觉token。不做量化，直接作为连续嵌入输入LLM。')
    P(doc,
        '分词器的总输出：62个力矩token + 8个触觉token + 1个视觉token = 71个"传感器token"，'
        '与文字token拼接后一起送入LLM。相当于LLM"看到"了一段由71个传感器词组成的"传感器语言"。')

    # ---- 图4：信号分词器详细结构 ----
    add_figure(doc, 'patent2_fig4_signal_tokenizer.png',
               '【图4：力-触觉信号分词器（Force-Tactile Tokenizer）网络结构详图】')

    P(doc, '分词器训练方式：', bold=True, indent=False)
    P(doc, '采用两阶段训练：')
    P(doc,
        '第一阶段——重建预训练（自监督）：'
        '训练目标是从量化后的token序列能还原回原始信号（重建损失）。'
        '类似VQ-VAE的训练方式：encoder输出连续特征→量化为离散token→decoder从token还原信号。'
        '损失 = 重建MSE + 向量量化commitment loss + codebook多样性正则。'
        '使用产线积累的所有拧紧数据（正常+异常均可，无需标注），约10万条力矩曲线。')
    P(doc,
        '第二阶段——诊断对齐微调（有监督）：'
        '冻结codebook和encoder大部分参数，对最后一层投影做LoRA微调，'
        '目标是让token序列被LLM正确理解（通过下游诊断准确率反向传播梯度到分词器最后一层）。'
        '使用约500条标注异常数据。')

    # ==== 模块二 ====
    P(doc, '模块二：领域微调多模态大模型（诊断推理核心）', bold=True, indent=False)
    P(doc,
        '采用7B参数量的多模态大语言模型（如InternVL2-7B）作为推理骨干。'
        '模型输入由三部分token序列拼接而成：')
    P(doc,
        '(1) 系统指令token（固定文本，约50个token）：定义角色、输出格式、安全约束等；'
        '(2) RAG检索到的历史案例token（约200-400个token）：最相似的前3条历史案例的诊断记录；'
        '(3) 当前传感器token（71个，来自模块一）+ 附加的结构化状态描述token'
        '（当前关节角度、目标力矩值、已旋转角度等数值信息，以文本形式，约30个token）。')
    P(doc,
        '模型输出为结构化JSON文本，包含：'
        'reasoning（推理过程，自然语言）、root_cause（根因类别，枚举值）、'
        'confidence（0-1）、direction（修正方向建议，定性）。')

    P(doc, 'RAG检索的嵌入模型（算法细节）：', bold=True, indent=False)
    P(doc,
        '案例检索不使用通用的sentence-transformer，而是使用力矩曲线对比学习训练的专用嵌入模型。'
        '该模型结构为分词器的encoder部分（冻结）+ 一个2层MLP投影头(256→128→64)。'
        '训练方法为监督对比学习（SupCon Loss）：')
    P(doc,
        '正样本对：同一异常类型的不同力矩曲线彼此为正；'
        '负样本对：不同异常类型的力矩曲线为负，其中重点挖掘"表面相似但原因不同"的困难负例'
        '（如交叉拧入和倾斜进入的力矩曲线形状相似但根因不同）。'
        '损失函数 L = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )，温度τ=0.07。'
        '训练完成后，案例库中每条案例的力矩曲线都通过该模型编码为64维嵌入向量。'
        '在线检索时，当前异常的力矩曲线同样编码为64维，做余弦相似度Top-3检索。')

    P(doc, 'LLM领域微调方式：', bold=True, indent=False)
    P(doc,
        '采用LoRA（rank=16, alpha=32）对LLM的注意力层Q/V矩阵做低秩适配。'
        '训练数据：约800条标注的异常案例，每条包含(传感器信号, 诊断JSON标注)。'
        '训练时将传感器token和文字token一起送入，目标为生成正确的诊断JSON。'
        '损失为标准的下一token预测交叉熵损失，仅在输出JSON部分计算（输入部分mask掉）。'
        '学习率2e-5，batch_size=8，训练约3个epoch（约30分钟单GPU）。')
    P(doc,
        '关键设计——传感器token的位置编码：'
        '为71个传感器token分配独立的可学习位置嵌入（不复用文字位置编码），'
        '且力矩token、触觉token、视觉token之间插入特殊分隔token [FORCE]、[TACTILE]、[VISION]，'
        '帮助LLM区分不同模态的信号来源。')

    # ==== 模块三 ====
    P(doc, '模块三：带几何约束的修正参数回归头（Correction Head）【核心算法创新之二】', bold=True, indent=False)
    P(doc,
        '传统做法是让LLM直接在文本中输出修正参数数字（如"右移1.5mm"），'
        '但LLM对精确数值的生成不够可靠。'
        '本方案在LLM骨干的最后一层隐状态上额外接一个参数回归头，'
        '直接从隐层表示中回归出精确修正参数，绕过LLM的文字生成环节。')
    P(doc, '回归头网络结构：')
    P(doc,
        '输入：LLM最后一层在[EOS] token位置的隐状态向量h∈R^4096'
        '（该位置汇聚了完整的推理上下文信息）。'
        '网络：3层MLP（4096→512→256→P），其中P为修正参数向量维度。')
    P(doc,
        '修正参数向量定义（P=8维）：\n'
        '  p[0]: 退出圈数 (范围0-3圈)\n'
        '  p[1]: 是否完全拔出 (0或1, sigmoid后取阈值)\n'
        '  p[2]: 角度补偿-俯仰(°) (范围-5°~+5°)\n'
        '  p[3]: 角度补偿-偏航(°) (范围-5°~+5°)\n'
        '  p[4]: 位置补偿-X(mm) (范围-2~+2mm)\n'
        '  p[5]: 位置补偿-Y(mm) (范围-2~+2mm)\n'
        '  p[6]: 重新拧入最大力矩(N·m)\n'
        '  p[7]: 重新拧入转速(rpm)')
    P(doc,
        '几何约束层：回归头的输出不是直接使用，而是经过一层约束变换：\n'
        '  - 各参数通过tanh激活映射到对应的物理允许范围（如角度±5°→tanh×5）；\n'
        '  - 位置补偿通过腕部相机的视觉测量结果做"soft gating"：\n'
        '    p_final[4:6] = σ(gate) × p_regression[4:6] + (1-σ(gate)) × p_visual[4:6]\n'
        '    其中p_visual来自视觉关键点检测的几何计算（精度±0.1mm），\n'
        '    gate是一个可学习的标量参数，训练中会学到视觉和回归结果之间的信任度平衡；\n'
        '  - 当LLM诊断的root_cause为"规格不匹配"时，所有修正参数置零'
        '（因为此时正确操作是放弃本次而非修正）。')

    P(doc, '回归头训练方式：', bold=True, indent=False)
    P(doc,
        '单独训练（不与LLM联合训练，避免破坏LLM推理能力）：')
    P(doc,
        '训练数据：从人工修正记录中提取——操作员手动修正异常时记录的动作参数作为GT标签。'
        '约800条修正记录，每条包含(LLM推理后的[EOS]隐状态，人工实际修正参数)。'
        '训练时冻结LLM全部参数（只用它提供隐状态特征），仅训练回归头MLP和gate参数。')
    P(doc,
        '损失函数：L = L_param + λ_range × L_range + λ_type × L_type\n'
        '  L_param = SmoothL1(p_pred, p_gt)，主损失，回归各参数；\n'
        '  L_range = ReLU(|p|−范围上限)²，惩罚超出物理范围的预测（作为软约束兜底）；\n'
        '  L_type = CrossEntropy(预测的修正类型, GT修正类型)——\n'
        '    额外加一个分类头预测修正类型'
        '（退出重对准/调角度/调位置/吹气/放弃），辅助回归头理解修正意图。'
        '  λ_range=0.1, λ_type=0.3。')

    # ==== 在线推理流程 ============
    P(doc, '在线推理流程', bold=True, indent=False)

    # ---- 图3：诊断工作流 ----
    add_figure(doc, 'patent2_fig3_diagnosis_workflow.png',
               '【图3：在线异常诊断与分级修正工作流程图】')

    P(doc,
        '当异常检测模型触发诊断后，完整推理流程如下：')
    P(doc,
        'Step 1（信号编码，~5ms）：'
        '将最近1秒的力矩和触觉信号送入分词器，得到71个传感器token。'
        '腕部图像经ViT提取视觉token。关节角度等数值信息编码为文字token。')
    P(doc,
        'Step 2（RAG检索，~10ms）：'
        '力矩信号经对比学习嵌入模型编码为64维向量，'
        '从案例库（预计算好的嵌入索引，支持FAISS加速）中检索Top-3相似案例。'
        '将案例的诊断记录文本作为上下文token拼入LLM输入。')
    P(doc,
        'Step 3（LLM推理，~1.5-2s）：'
        '全部token拼接送入LLM，生成诊断JSON（含reasoning、root_cause、confidence）。'
        '同时保存[EOS]位置的隐状态向量h。')
    P(doc,
        'Step 4（参数回归，~2ms）：'
        '将h送入修正参数回归头，输出8维修正参数向量。'
        '经几何约束层处理后得到最终修正参数。')
    P(doc,
        'Step 5（安全分级+执行）：'
        '根据confidence值和修正参数幅度进行安全分级：\n'
        '  低风险（直接执行）：仅退出动作（p[0]>0且p[2:6]全为0）；\n'
        '  中风险（3秒默认确认）：角度补偿<3°且位置补偿<0.5mm；\n'
        '  高风险（必须确认）：补偿量大或confidence<0.6或root_cause=规格不匹配。\n'
        '通过安全分级后，将修正参数转化为机器人动作指令执行。')
    P(doc,
        'Step 6（验证+经验积累）：'
        '修正后以正常流程重新拧入验证——自由旋入扭矩正常+最终力矩达标则通过。'
        '完整记录存入案例库，定期增量LoRA微调LLM和回归头。')

    P(doc, '总推理延迟：~1.6-2.1秒（主要耗时在LLM token生成）。')

    # ---- 关键步骤总结 ----
    P(doc, '关键步骤总结', bold=True, indent=False)
    P(doc,
        '(1) 力-触觉信号分词器（模块一）：通过因果卷积+向量量化将连续传感信号转化为离散token，'
        '使LLM能"阅读"传感器信号。相比模板文字描述，端到端训练的分词器能捕捉人工无法设计的细微特征。'
        '这是解决"传感器信号如何输入大模型"的算法创新。')
    P(doc,
        '(2) 对比学习训练的案例检索嵌入（模块二子模块）：'
        '相比通用文本嵌入，力矩曲线专用嵌入在异常检索上的召回精度大幅提升。'
        '困难负例挖掘确保了"表面相似但原因不同"的案例能被正确区分。')
    P(doc,
        '(3) 带几何约束的修正参数回归头（模块三）：'
        '从LLM隐状态直接回归修正参数，绕过LLM文本生成的数值不精确问题。'
        '视觉-回归soft gating机制让系统自动学习两种精度来源之间的最优融合权重。'
        '几何范围约束层保证输出参数始终在物理可行域内。')
    P(doc,
        '(4) LLM思维链推理提供可解释的诊断过程：'
        '与纯黑盒分类器不同，LLM输出的reasoning字段详细记录了'
        '"看到了什么→排除了什么→得出什么结论"的完整推理链，可供人工审查。')
    P(doc,
        '(5) 安全分级机制保障工业可部署性：'
        '低/中/高三级管控确保系统在有把握时果断自主修正、'
        '没把握时谨慎请示，适配真实产线的安全要求。')

    # ============ 2.4 有益效果 ============
    H(doc, '2.4 有益效果')
    P(doc,
        '(1) 大幅减少异常停机导致的产能损失。'
        '传统系统异常后等待人工处理平均3-5分钟/次，'
        '本方案对可自主修正的异常（约占总异常60-70%）处理时间约8-20秒'
        '（含1.5-2.5秒诊断+2-5秒修正+2-5秒验证+安全确认窗口），'
        '产能恢复速度提升约10-20倍。按每天30次异常计算，'
        '每天可节省约1.5-2小时的停机等待时间。')
    P(doc,
        '(2) 诊断结论透明可解释。'
        '整个推理过程以自然语言输出（包括"为什么排除了其他可能性"），'
        '质量工程师可以直接阅读审查。这是RL黑盒策略或简单分类器无法提供的。'
        '便于工艺问题的根因追溯和系统性改进。')
    P(doc,
        '(3) 对新异常类型具有零样本适应能力。'
        '大模型基于对物理世界的通用理解进行推理，'
        '结合RAG中相似（但不完全相同）的历史案例做类比推断，'
        '即使遇到训练集中没有的新异常模式，也能给出合理的诊断猜测'
        '（此时confidence会偏低，触发人工确认模式，既不冒险也不放弃诊断）。')
    P(doc,
        '(4) 系统持续自进化。'
        '每次处理（无论成功或转人工）都在积累经验。'
        '运行半年后，案例库覆盖度可达90%以上的常见异常模式，'
        '诊断正确率从初始的约70%逐步提升至90%+。')
    P(doc,
        '(5) 降低对经验丰富操作员的依赖。'
        '传统方案依赖老师傅凭手感和经验判断异常原因，'
        '本方案将诊断经验系统化沉淀为案例库和模型参数，'
        '新操作员通过阅读系统输出的诊断报告也能快速理解异常原因并学习处理方法。')
    P(doc,
        '(6) 安全分级机制兼顾自主性与安全性。'
        '不是"什么都让机器自己来"的冒险方案，也不是"出了问题就停下等人"的保守方案，'
        '而是根据置信度和风险等级灵活决策——该果断时果断、该谨慎时谨慎。')

    # ============ 附件 ============
    H(doc, '附件')
    P(doc, '参考文献：', bold=True, indent=False)
    P(doc,
        '[1] Physical Intelligence, "pi0: VLA Flow Model for General Robot Control", 2024. '
        '[2] Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", CoRL 2024. '
        '[3] Stanford, "FuSe: Fusing Sensory Data for Robot Learning through Language", CoRL 2024. '
        '[4] Meta FAIR, "Sparsh: Self-supervised Touch Representations for Vision-Based Tactile Sensing", 2024. '
        '[5] BMW/Fraunhofer, "DL-Based Quality Prediction for Robotic Fastener Assembly Operations", RCIM 2024. '
        '[6] ByteDance, "GR-2: Generative Video-Language-Action Model", 2024. '
        '[7] Toyota Research, "Force-Torque Guided Robotic Bolt Tightening Using Deep RL", RA-L 2024. '
        '[8] Zheng et al., "OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Manipulation", 2026. '
        '[9] Xue et al., "Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning", 2025. '
        '[10] Ruan et al., "ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer", 2026. '
        '[11] Heng et al., "ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Manipulation", 2025. '
        '[12] Hansen et al., "TD-MPC2: Scalable Robust World Models for Continuous Control", ICML 2024.')

    # ---- 保存 ----
    out = '/home/chenshuai/Project/TactileACT-cs/paper/patent/patent2_final_with_figures.docx'
    doc.save(out)
    print(f'Patent saved to: {out}')


if __name__ == '__main__':
    build()
