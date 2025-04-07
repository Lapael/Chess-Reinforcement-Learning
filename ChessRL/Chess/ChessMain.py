import pygame as p
from Chess import ChessEngine, SmartMoveFinder
from multiprocessing import Process, Queue

p.display.set_caption('Chess By EUNSEONG')
p.display.set_icon(p.image.load('chess/images/wK.png'))
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
        IMAGES[piece] = p.transform.scale(p.image.load('chess/images/' + piece + '.png'), (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color('white'))
    moveLogFont = p.font.SysFont('Arial', 12, False, False)
    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    moveMade = False
    animate = False
    loadImages()
    running = True
    sqSelected = ()  # 이동 칸 선택
    playerClicks = []  # 이동 칸 저장
    gameOver = False
    playerOne = True  # 백 True : 인간 / False : AI
    playerTwo = False  # 흑 True : 인간 / False : AI
    AIThinking = False
    moveFinderProcess = False
    moveUndone = False

    while running:
        humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)

        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            # 마우스 조작
            elif e.type == p.MOUSEBUTTONDOWN:
                if not gameOver:
                    location = p.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    if sqSelected == (row, col) or col >= 8:
                        sqSelected = ()
                        playerClicks = []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)
                    if len(playerClicks) == 2 and humanTurn:
                        move = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        # print(move.getChessNotation())
                        for i in range(len(validMoves)):
                            if move == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                animate = True
                                sqSelected = ()  # 선택 리셋
                                playerClicks = []
                        if not moveMade:
                            playerClicks = [sqSelected]


            # 키보드 조작
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:
                    gs.undoMove()
                    moveMade = True
                    animate = False
                    gameOver = False
                    if AIThinking:
                        moveFinderProcess.terminate()
                        AIThinking = False
                    moveUndone = True

                if e.key == p.K_r:
                    gs = ChessEngine.GameState()
                    validMoves = gs.getValidMoves()
                    sqSelected = ()
                    playerClicks = []
                    moveMade = False
                    animate = False
                    gameOver = False
                    if AIThinking:
                        moveFinderProcess.terminate()
                        AIThinking = False
                    moveUndone = True

        # AI
        if not gameOver and not humanTurn and not moveUndone:
            if not AIThinking:
                AIThinking = True
                print('thinking...')
                returnQueue = Queue() # thread 공유하는 방법
                moveFinderProcess = Process(target=SmartMoveFinder.findBestMove, args=(gs, validMoves, returnQueue))
                moveFinderProcess.start() # AI 호출

            if not moveFinderProcess.is_alive():
                print('done thinking')
                AIMove = returnQueue.get()
                if AIMove is None:
                    AIMove = SmartMoveFinder.findRandomMove(validMoves)
                gs.makeMove(AIMove)
                moveMade = True
                animate = True
                AIThinking = False

        if moveMade:
            if animate:
                animateMove(gs.moveLog[-1], screen, gs.board, clock)
            validMoves = gs.getValidMoves()
            moveMade = False
            animate = False
            moveUndone = False

        drawGameState(screen, gs, validMoves, sqSelected, moveLogFont)

        if gs.checkMate or gs.staleMate:
            gameOver = True
            text = 'Stalemate' if gs.staleMate else 'Black wins by checkmate' if gs.whiteToMove else 'White wins by checkmate'
            drawEndGameText(screen, text)

        clock.tick(MAX_FPS)
        p.display.flip()

def drawGameState(screen, gs, validMoves, sqSelected, moveLogFont):
    drawBoard(screen)  # 사각형
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)  # 기물
    drawMoveLog(screen, gs, moveLogFont)

def drawBoard(screen):
    global colors
    colors = [color_light, color_dark]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# 선택 사각형 하이라이트
def highlightSquares(screen, gs, validMoves, sqSelected):
    colors_sel = [light_stress, dark_stress]
    colors_can = [light_canMove, dark_canMove]
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):  # 선택된 칸이 움직일 수 있음
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.fill(colors_sel[(r + c) % 2])
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))

            for move in validMoves:
                s.fill(colors_can[(move.endRow + move.endCol) % 2])
                if move.startRow == r and move.startCol == c:
                    if gs.board[move.endRow][move.endCol] == '--' and not move.isEnpassantMove:
                        radius = SQ_SIZE // 5.5
                        circleSurface = p.Surface((radius * 2, radius * 2), p.SRCALPHA)
                        p.draw.circle(circleSurface, colors_can[(move.endRow + move.endCol) % 2], (radius, radius),
                                      radius)
                        screen.blit(circleSurface, (move.endCol * SQ_SIZE + (SQ_SIZE - radius * 2) // 2,
                                                    move.endRow * SQ_SIZE + (SQ_SIZE - radius * 2) // 2))
                    else:
                        circleSurface = p.Surface((SQ_SIZE, SQ_SIZE), p.SRCALPHA)
                        p.draw.circle(circleSurface, colors_can[(move.endRow + move.endCol) % 2],
                                      (SQ_SIZE / 2, SQ_SIZE / 2), SQ_SIZE / 2, 6)
                        screen.blit(circleSurface, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))


def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]  # r행c열
            if piece != '--':
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def drawMoveLog(screen, gs, font):
    moveLogRect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color('Black'), moveLogRect)

    moveLog = gs.moveLog
    moveTexts = []

    # 체스 기보 형식으로 변환
    for i in range(0, len(moveLog), 2):
        moveString = str(i // 2 + 1) + ". " + str(moveLog[i]) + " "
        if i + 1 < len(moveLog):  # 흑 이동
            moveString += str(moveLog[i + 1]) + " "
        moveTexts.append(moveString)

    movesPerRow = 3  # 한 줄에 3개씩 배치
    padding = 5  # 왼쪽 여백
    lineSpacing = 2  # 줄 간격
    textY = padding  # 초기 Y 위치

    columnWidth = MOVE_LOG_PANEL_WIDTH // movesPerRow  # 열 너비 계산

    for i in range(0, len(moveTexts), movesPerRow):
        for j in range(movesPerRow):
            if i + j < len(moveTexts):
                textObject = font.render(moveTexts[i + j], True, p.Color('White'))
                textX = moveLogRect.x + padding + j * columnWidth  # X 좌표 계산
                textLocation = (textX, textY)
                screen.blit(textObject, textLocation)

        textY += textObject.get_height() + lineSpacing  # 다음 줄로 이동


# 애니메이션
def animateMove(move, screen, board, clock):
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 1  # 속도 조절 변수 : 클수록 느림
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
    font = p.font.SysFont(None, 32, False, False)
    textObject = font.render(text, 0, p.Color('White'))
    textWidth = textObject.get_width()
    textHeight = textObject.get_height()
    rectWidth = textWidth + 40  # 텍스트 너비에 여백 추가
    rectHeight = textHeight + 20  # 텍스트 높이에 여백 추가
    rectX = (BOARD_WIDTH - rectWidth) / 2
    rectY = (BOARD_HEIGHT - rectHeight) / 2

    textLocation = p.Rect(rectX, rectY, rectWidth, rectHeight)
    border_radius = 10
    p.draw.rect(screen, (60, 57, 56), textLocation, border_radius=border_radius)

    # 텍스트 그리기
    textPosX = rectX + (rectWidth - textWidth) / 2
    textPosY = rectY + (rectHeight - textHeight) / 2
    screen.blit(textObject, (textPosX, textPosY))


if __name__ == '__main__':
    main()
