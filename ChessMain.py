import pygame as p
import ChessEngine, RL_Agent
import time

p.display.set_caption('Chess By EUNSEONG')
p.display.set_icon(p.image.load('images/wK.png'))
BOARD_WIDTH = BOARD_HEIGHT = 512  # 크기
MOVE_LOG_PANEL_WIDTH = 256
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
DIMENSION = 8  # 8*8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES = {}

color_light = (235, 236, 210)
color_dark = (115, 149, 87)
light_stress = (245, 245, 147)  # 흰 칸 선택 시 강조 색
dark_stress = (185, 201, 85)  # 검은 칸 선택 시 강조 색
light_canMove = (202, 203, 179)
dark_canMove = (100, 128, 73)


def loadImages():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load('images/' + piece + '.png'), (SQ_SIZE, SQ_SIZE))

def training_loop(num_episodes): # Main 학습 루프

    agent = RL_Agent.Agent()
    memory = [] 
    BATCH_SIZE = 2048 # 2048개의 게임 데이터가 모이면 학습
    
    visualize_next_game = False
    
    for episode in range(num_episodes):
        print(f"--- Episode {episode + 1} ---")

        # 시각화 여부 결정
        if visualize_next_game: # or (episode + 1) % 10 == 0:
            game_history, visualize_next_game = run_visual_game(agent)
        else:
            game_history = run_silent_game(agent)

        # 결과 출력 및 데이터 저장
        if game_history:
            result = game_history[-1][-1] # 마지막 데이터의 보상 값으로 결과 확인
            print(f"Game Over. Result: \033[31m{result}\033[0m. States collected: {len(game_history)}")
            memory.extend(game_history)
        
        # 일정양의 데이터가 모이면 학습
        if len(memory) >= BATCH_SIZE:
            print(f"--- Training with {len(memory)} states ---")
            agent.train_step(memory)
            memory = [] # 데이터 초기화

        if (episode + 1) % 100 == 0:
            agent.save_model(f"chess_agent_model_{episode+1}.pth")

    print("Training finished.")
    agent.save_model("chess_agent_model.pth")



def run_visual_game(agent): # 시각화 학습 모드

    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    loadImages()
    
    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    game_history = []
    gameOver = False
    move_count = 0

    while not gameOver:
        for e in p.event.get():
            if e.type == p.QUIT:
                return None, False # 강종

        state_tensor = gs.state_to_tensor() # 보드 상태 텐서로 변환
        
        # 학습 시 수순에 따라 온도 조절
        temperature = 1.0 if move_count < 30 else 0.0
        AIMove, mcts_policy = agent.select_action(gs, validMoves, temperature)
        
        if AIMove:
            game_history.append([state_tensor, mcts_policy]) # 상태와 MCTS 정책 저장
            gs.makeMove(AIMove)
            move_count += 1

            validMoves = gs.getValidMoves()
            animateMove(gs.moveLog[-1], screen, gs.board, clock)
        
        drawGameState(screen, gs, validMoves, (), p.font.SysFont('Arial', 12))
        
        # 게임 종료 조건 확인
        if not validMoves or gs.checkMate or gs.staleMate or gs.insufficientMaterial or gs.threefold_repetition:
            gameOver = True

        p.display.flip()
        clock.tick(MAX_FPS)

    # 게임 종료 후 결과 처리 및 보상
    final_reward, text = get_reward_and_text(gs)
    assign_rewards(game_history, final_reward)
    
    # 결과 화면 표시
    drawEndGameText(screen, text)
    p.display.flip()
    
    # visualize_next = wait_for_visualization_input(clock) # 쓸모 없을듯? 지금은
    visualize_next = False
    
    p.quit()
    return game_history, visualize_next

def run_silent_game(agent): # 비시각화 학습 모드 - GPU 절약 ㅆㄱㄴ

    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    game_history = []
    move_count = 0
    
    while not (gs.checkMate or gs.staleMate or gs.insufficientMaterial or gs.threefold_repetition):
        state_tensor = gs.state_to_tensor()
        
        # 학습 시 수순에 따라 온도 조절
        temperature = 1.0 if move_count < 30 else 0.0
        AIMove, mcts_policy = agent.select_action(gs, validMoves, temperature)
        
        if not AIMove:
            break # 가능한 수가 없으면 루프 종료
            
        game_history.append([state_tensor, mcts_policy]) # 상태와 MCTS 정책 저장
        gs.makeMove(AIMove)
        move_count += 1

        validMoves = gs.getValidMoves()

    final_reward, _ = get_reward_and_text(gs)
    assign_rewards(game_history, final_reward)
    
    return game_history

def assign_rewards(game_history, reward): # 최종 보상 산출

    for i in range(len(game_history)):
        # 백의 승리(reward=1)일 때, 백의 턴이었던 상태는 +1, 흑의 턴은 -1
        # 흑의 승리(reward=-1)일 때, 백의 턴이었던 상태는 -1, 흑의 턴은 +1
        # 텐서의 12번 채널(턴 정보)을 확인: 1이면 백의 턴, 0이면 흑의 턴
        turn = game_history[i][0][12, 0, 0] 
        if turn == 1: # 백턴
            game_history[i].append(reward)
        else: # 흑턴
            game_history[i].append(-reward)

def get_reward_and_text(gs):

    if gs.checkMate:
        reward = -1 if gs.whiteToMove else 1
        text = 'Black wins by checkmate' if gs.whiteToMove else 'White wins by checkmate'
    elif gs.staleMate or gs.insufficientMaterial or gs.threefold_repetition:
        reward = -0.1 # 무승부를 약한 패배로
        if gs.staleMate:
            text = 'Draw by Stalemate'
        elif gs.insufficientMaterial:
            text = 'Draw by Insufficient Material'
        else:
            text = 'Draw by Three-fold Repetition'
    else: # 게임이 아직 안 끝난 경우 (이론상으론 없어야 함)
        reward = 0
        text = 'Game in progress'
    return reward, text

# def wait_for_visualization_input(clock): # 다음 게임 시각화 선택 - 비활성화됨

#     end_time = p.time.get_ticks() + 3000
#     while p.time.get_ticks() < end_time:
#         for e in p.event.get():
#             if e.type == p.QUIT:
#                 return False
#             if e.type == p.KEYDOWN:
#                 if e.key == p.K_v:
#                     print("Next game will be visualized.")
#                     return True
#         clock.tick(MAX_FPS)
#     print("Continuing with silent training.")
#     return False

'''--- 기존 체스 게임 함수들 ---'''

def drawGameState(screen, gs, validMoves, sqSelected, moveLogFont):
    drawBoard(screen)
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)
    drawMoveLog(screen, gs, moveLogFont)

def drawBoard(screen):
    global colors
    colors = [color_light, color_dark]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            colors_sel = [light_stress, dark_stress]
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.fill(colors_sel[(r + c) % 2])
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
            
            colors_can = [light_canMove, dark_canMove]
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    if gs.board[move.endRow][move.endCol] == '--' and not move.isEnpassantMove:
                        radius = SQ_SIZE // 5.5
                        circleSurface = p.Surface((radius * 2, radius * 2), p.SRCALPHA)
                        p.draw.circle(circleSurface, colors_can[(move.endRow + move.endCol) % 2], (radius, radius), radius)
                        screen.blit(circleSurface, (move.endCol * SQ_SIZE + (SQ_SIZE - radius * 2) // 2, move.endRow * SQ_SIZE + (SQ_SIZE - radius * 2) // 2))
                    else:
                        circleSurface = p.Surface((SQ_SIZE, SQ_SIZE), p.SRCALPHA)
                        p.draw.circle(circleSurface, colors_can[(move.endRow + move.endCol) % 2], (SQ_SIZE / 2, SQ_SIZE / 2), SQ_SIZE / 2, 6)
                        screen.blit(circleSurface, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != '--':
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawMoveLog(screen, gs, font):
    moveLogRect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color('Black'), moveLogRect)
    moveLog = gs.moveLog
    moveTexts = []
    for i in range(0, len(moveLog), 2):
        moveString = str(i // 2 + 1) + ". " + str(moveLog[i]) + " "
        if i + 1 < len(moveLog):
            moveString += str(moveLog[i + 1]) + " "
        moveTexts.append(moveString)
    
    movesPerRow = 3
    padding = 5
    lineSpacing = 2
    textY = padding
    columnWidth = MOVE_LOG_PANEL_WIDTH // movesPerRow
    for i in range(0, len(moveTexts), movesPerRow):
        for j in range(movesPerRow):
            if i + j < len(moveTexts):
                textObject = font.render(moveTexts[i + j], True, p.Color('White'))
                textX = moveLogRect.x + padding + j * columnWidth
                textLocation = (textX, textY)
                screen.blit(textObject, textLocation)
        textY += textObject.get_height() + lineSpacing

def animateMove(move, screen, board, clock):
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 3
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR * frame / frameCount, move.startCol + dC * frame / frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        color = colors[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        if move.pieceCaptured != '--':
            if move.isEnpassantMove:
                enpassantRow = move.endRow + 1 if move.pieceCaptured[0] == 'b' else move.endRow - 1
                endSquare = p.Rect(move.endCol * SQ_SIZE, enpassantRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            screen.blit(IMAGES[move.pieceCaptured], endSquare)
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)

def drawEndGameText(screen, text):
    font = p.font.SysFont(None, 48, True, True)
    textObject = font.render(text, 0, p.Color('White'))
    textWidth = textObject.get_width()
    textHeight = textObject.get_height()
    rectWidth = textWidth + 40
    rectHeight = textHeight + 20
    rectX = (BOARD_WIDTH - rectWidth) / 2
    rectY = (BOARD_HEIGHT - rectHeight) / 2
    textLocation = p.Rect(rectX, rectY, rectWidth, rectHeight)
    border_radius = 10
    p.draw.rect(screen, (60, 57, 56), textLocation, border_radius=border_radius)
    textPosX = rectX + (rectWidth - textWidth) / 2
    textPosY = rectY + (rectHeight - textHeight) / 2
    screen.blit(textObject, (textPosX, textPosY))

'''--- 기존 체스 게임 함수들 여기까지 ---'''

def play_vs_ai(agent, human_is_white): # 모델 불러와서 체스 무기

    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    loadImages()
    
    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    gameOver = False
    sqSelected = ()
    playerClicks = []

    while not gameOver:
        is_human_turn = (gs.whiteToMove and human_is_white) or (not gs.whiteToMove and not human_is_white)

        if is_human_turn:
            for e in p.event.get():
                if e.type == p.QUIT:
                    gameOver = True
                    break
                
                if e.type == p.MOUSEBUTTONDOWN:
                    location = p.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    
                    if sqSelected == (row, col) or col >= 8:
                        sqSelected = ()
                        playerClicks = []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)
                    
                    if len(playerClicks) == 2:
                        move = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        move_made = False
                        for i in range(len(validMoves)):
                            if move == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                move_made = True
                                validMoves = gs.getValidMoves()
                                sqSelected = ()
                                playerClicks = []
                                break
                        if not move_made:
                            playerClicks = [sqSelected]

        else: # AI 턴
            for e in p.event.get():
                if e.type == p.QUIT:
                    gameOver = True
                    break
            
            if not gameOver:
                AIMove, _ = agent.select_action(gs, validMoves, temperature=0)
                if AIMove:
                    gs.makeMove(AIMove)
                    validMoves = gs.getValidMoves()

        drawGameState(screen, gs, validMoves, sqSelected, p.font.SysFont('Arial', 12, False, False))
        
        # 게임 종료 조건의 확인
        if gs.checkMate or gs.staleMate or gs.insufficientMaterial or gs.threefold_repetition:
            gameOver = True
            _, text = get_reward_and_text(gs)
            drawEndGameText(screen, text)

        p.display.flip()
        clock.tick(MAX_FPS)

    p.quit()

if __name__ == '__main__':

    play_mode = True

    if not play_mode:
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        training_loop(num_episodes=2000)
        print(f"{start_time} ~ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    elif play_mode:
        agent = RL_Agent.Agent()
        agent.load_model("chess_agent_model_500.pth")
        play_vs_ai(agent, human_is_white=1557)
