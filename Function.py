import PySimpleGUI as sg
import cv2
from modelAPI import emo_model
import random
from emoDictionary import emo_lt, reverse_emo
from datetime import datetime


def img_gui():
    """这个函数是设计识别图片的图形用户界面"""
    emo_list, emo_trans = emo_lt()
    rand_emo = random.choice(emo_list)
    rand_emo_cn = emo_trans[rand_emo]

    layout = [
        [sg.Text("在本地图库中上传表情图片.", font=("微软雅黑", 14, "bold"), justification="center", expand_x=True)],
        [sg.VPush()],
        [sg.HorizontalSeparator()],
        [sg.Text(f"请上传与 {rand_emo_cn} 表情相反的图片。", font=("微软雅黑", 11, "bold"))],
        [sg.Text("请输入图片路径：", font=("微软雅黑", 8, "bold")), sg.InputText(key="path")],
        [sg.Frame("图片预览", [[sg.Image(key="image_preview", size=(380, 380))]])],
        [sg.VPush()],
        [sg.HorizontalSeparator()],
        [sg.Text("此处将显示结果", font=("微软雅黑", 11, "bold"), key="res_text")],
        [sg.Text("", key="game_text")],
        # [sg.Column([[sg.Button("识别")]], justification='left', expand_x=True),
        #  sg.Column([[sg.Button("帮助")]], justification='center', expand_x=True),
        #  sg.Column([[sg.Button("退出")]], justification='right', expand_x=True)],       # 这种方法不能对齐
        [sg.Button("识别"), sg.Push(), sg.Button("帮助"), sg.Push(), sg.Button("退出")],

    ]

    window = sg.Window("在图片中识别表情", layout, resizable=True)

    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
        if event == "帮助":
            help_img()
        if event == "识别":
            img_path = values["path"]
            img = cv2.imread(img_path)
            if img is None:
                sg.popup("图片路径输入有误！", title="错误")
            else:
                resized_img = cv2.resize(img, (380, 380))
                img_bytes = cv2.imencode('.png', resized_img)[1].tobytes()
                window["image_preview"].update(data=img_bytes)

                res = emo_model(img_path)
                main_emo = res[0]
                conf = res[1]
                res_text = f"画面中的人物表情有 {conf} 的概率是 {emo_trans[main_emo]}。"
                if reverse_emo(rand_emo, main_emo):
                    game_text = f"你赢了！因为 {emo_trans[main_emo]} 和 {rand_emo_cn} 是相反的表情。"
                    window["game_text"].update(game_text)

                    now = datetime.now()
                    now = str(now)
                    with open('_EmoLog.txt', "a", encoding="utf-8") as file:        # 写入日志
                        file.write(now + '\n' + res_text + '\n' + game_text + '\n \n')

                else:
                    game_text = f"你失败了！因为图片中人物表情和 {rand_emo_cn} 并不相反。"
                    window["game_text"].update(game_text)

                    now = datetime.now()
                    now = str(now)
                    with open('_EmoLog.txt', "a", encoding="utf-8") as file:        # 写入日志
                        file.write(now + '\n' + res_text + '\n' + game_text + '\n \n')

                # cv2.imshow("emotion img", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                window["res_text"].update(res_text)

    window.close()


def help_img():
    """这个函数设置图片界面中的帮助文档"""
    layout = [[sg.Text("在电脑资源管理器中选择表情图片，然后输入图片路径（路径不需要用引号包裹），\n"
                       "点击进行识别，即可调用模型识别出图片人物的表情。(～￣▽￣)～")],
              [sg.Column([[sg.Button("退出")]], justification='center',
                         expand_x=True, element_justification='center')]]

    window = sg.Window("帮助界面", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
    window.close()


def cap_gui():
    """这个函数是设计识别摄像头的图形用户界面"""
    emo_list, emo_trans = emo_lt()
    rand_emo = random.choice(emo_list)
    layout = [
        [sg.Text("打开摄像头捕获表情.", font=("微软雅黑", 14, "bold"), justification='center', expand_x=True)],
        [sg.VPush()],
        [sg.HorizontalSeparator()],
        [sg.Text(f"请在摄像头前做出和 {emo_trans[rand_emo]} 相反的表情。", font=("微软雅黑", 11, "bold"))],
        # [sg.Text(f"请在摄像头前做出和演示图片相反的表情。", font=("微软雅黑", 11, "bold"))],
        [sg.Button("打开摄像头", expand_x=True)],
        [sg.Frame("图片演示/预览", [[sg.Image(key="image_preview", size=(380, 285))]])],
        [sg.Text("此处将显示结果", font=("微软雅黑", 11), key="res_text")],
        [sg.Text("", font=("微软雅黑", 11), key="game_text")],
        [sg.VPush()],
        [sg.HorizontalSeparator()],
        [sg.Button("捕获(s)"), sg.Push(), sg.Button("帮助"), sg.Push(), sg.Button("退出")],
    ]
    window = sg.Window("在摄像头中识别表情", layout, size=(430, 565), resizable=True)

    cap = None
    is_capturing = False
    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED or event == "退出":
            break
        if event == "帮助":
            help_cap()
        if event == "捕获(s)":
            if cap is None:
                sg.popup("请先打开摄像头！", title="错误")
                # print("断点1：打开摄像头")
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    sg.popup("无法获取摄像头画面！", title="错误")
                    continue
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                        print("断点2：捕获退出")
                        break
                    res = emo_model(frame)
                    resize_frame = cv2.resize(frame, (380, 285))
                    img_bytes = cv2.imencode(".png", resize_frame)[1].tobytes()
                    window["image_preview"].update(data=img_bytes)
                    main_emo = res[0]
                    conf = res[1]
                    res_text = f"此时你的表情有{conf}的概率是{emo_trans[main_emo]}"
                    window["res_text"].update(res_text)
                    if reverse_emo(rand_emo, main_emo):
                        game_text = f"你赢了！因为 {emo_trans[main_emo]} 和 {emo_trans[rand_emo]} 是相反的表情。"
                    else:
                        game_text = f"你失败了！因为画面中人物表情和 {emo_trans[rand_emo]} 并不相反。"
                    window["game_text"].update(game_text)

                    now = datetime.now()
                    now = str(now)
                    with open('_EmoLog.txt', "a", encoding="utf-8") as file:
                        file.write(now + '\n' + res_text + '\n' + game_text + '\n \n')

        if event == "打开摄像头":
            # print("断点3，开摄像头")
            if cap is None or not cap.isOpened():       # 嵌套乱了，所以尝试这种写法
                cap = cv2.VideoCapture(0)
                is_capturing = True

        if is_capturing and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Emotion Capture", frame)  # 循环点有问题
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                is_capturing = False
                cap.release()
                cv2.destroyAllWindows()
            try:
                if cv2.getWindowProperty("Emotion Capture", cv2.WND_PROP_VISIBLE) < 1:
                    is_capturing = False
                    cap.release()
                    cv2.destroyAllWindows()
            except:
                pass

    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    window.close()


def help_cap():
    layout = [[sg.Text("按下“打开摄像头”，可调用电脑默认摄像头进行画面捕获，\n"
                       "键入“s”键可捕获当前画面，键入“q”键可退出捕获。")],
              [sg.Column([[sg.Button("退出")]], justification='center',
                         expand_x=True, element_justification='center')]]
    window = sg.Window("帮助界面", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
    window.close()


def vid_gui():
    layout = [[sg.Text("对本事视频进行表情识别.")],
              [sg.Text("输入视频地址："), sg.InputText(key="path")],
              [sg.Column([[sg.Button("开始识别")]], justification='left', expand_x=True),
               sg.Column([[sg.Button("帮助")]], justification='center', expand_x=True),
               sg.Column([[sg.Button("退出")]], justification='right', expand_x=True)]]
    window = sg.Window("在视频中识别表情", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
        if event == "帮助":
            help_vid()
        if event == "开始识别":
            vid_path = values["path"]
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                sg.popup("视频打开失败！", title="错误")
                break       # 这块把while整个循环给退了，算了就这样吧
            cv2.namedWindow("Video Emotion Recognition.", cv2.WINDOW_NORMAL)

            while True:
                ret, frame = cap.read()
                if not ret:  # 这个判断是防止视频播放完后程序异常报错！
                    sg.popup("视频已播放完毕！", title="提示")
                    break
                res = emo_model(frame)
                main_emo = res[0]
                cv2.putText(frame, f"{main_emo}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Video Emotion Recognition.", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        now = datetime.now()
        now = str(now)
        with open('_EmoLog.txt', "a", encoding="utf-8") as file:
            file.write(now + '上传了一个视频进行识别。')

    window.close()


def help_vid():
    layout = [[sg.Text("请输入视频路径，然后点击识别，即可调用模型识别出视频人物的表情。\n"
                       "输入的路径不要有引号。"
                       "视频中人物表情会实时地显示在画面左上角中。")],
              [sg.Column([[sg.Button("退出")]], justification='center',
                         expand_x=True, element_justification='center')]]
    window = sg.Window("帮助界面", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
    window.close()


def main_gui():
    """这个是程序的主界面，这里调用了之前定义的三个主要功能函数"""
    sg.theme('DarkTeal9')
    sg.set_options(font=("微软雅黑", 12))
    layout = [
        [sg.Text("欢迎使用反向表情人机交互游戏！", font=("微软雅黑", 16, 'bold'), justification='center')],
        [sg.VPush()],
        [sg.HorizontalSeparator()],
        [sg.Text("请选择识别方式：", font=("微软雅黑", 10, 'bold'), justification='center')],
        [sg.Button("打开摄像头识别", expand_x=True, size=(15, 1), button_color=('white', '#0078D7'))],
        [sg.Button("打开本地图片识别", expand_x=True, size=(15, 1), button_color=('white', '#0078D7'))],
        [sg.Button("打开本地视频识别", expand_x=True, size=(15, 1), button_color=('white', '#0078D7'))],
        [sg.HorizontalSeparator()],
        [sg.Button("帮助"), sg.Push(), sg.Button("作者信息"), sg.Push(),
         sg.Button("退出", button_color=('white', '#E81123'))]
    ]
    window = sg.Window("游戏界面", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
        if event == "帮助":
            help_main()
        if event == "作者信息":
            sg.popup("\t作者：Yang.Fan\n"
                     "\t学号：2102140402046\n"
                     "\t邮箱：a2078769713@sina.com\n"
                     "\t主页：https://github.com/GunsK-III\t", title="作者信息")
        if event == "打开摄像头识别":
            cap_gui()
        if event == "打开本地图片识别":
            img_gui()
        if event == "打开本地视频识别":
            vid_gui()
    window.close()


def help_main():
    """这是主界面中的帮助文档"""
    layout = [[sg.Text("游戏核心玩法：\n"
                       "      该模型可识别五种表情，分别是开心、伤心、愤怒、平静、惊讶。\n"
                       "      我们定义开心和伤心/愤怒相对，平静和惊讶相对。\n"
                       "      当计算机展示出一种表情的相关元素时，玩家需要做出或上传相反的表情，即可获胜。\n"
                       "      比如，当计算机展示愤怒的相关元素时，玩家需要做出开心的表情，即可获胜。\n"
                       "本系统提供三种识别方式，\n"
                       "1.打开摄像头识别。\n"
                       "      计算机已链接到摄像头时，可使用该方法进行游戏。\n"
                       "      计算机会随机展示一种表情，玩家及时在摄像头前做出相反的表情可获胜。\n"
                       "2.打开本地图片识别，此时可打开电脑资源管理器，\n"
                       "      如果计算机没有链接摄像头，可使用该方式进行游戏。\n"
                       "      计算机展示一种表情之后，玩家在本地选择并读取一张相反的表情图片可获胜。\n"
                       "3.打开本地视频识别，该方式可以识别视频中人物的表情\n"
                       "在二级界面中点击“帮助”按钮可查看对应功能的详细使用方法。(～￣▽￣)～")],
              [sg.Column([[sg.Button("退出")]], justification='center',
                         expand_x=True, element_justification='center')]]
    window = sg.Window("帮助界面", layout, resizable=True)
    while True:
        event, values = window.read()
        if event == "退出" or event == sg.WIN_CLOSED:
            break
    window.close()


if __name__ == '__main__':
    main_gui()
