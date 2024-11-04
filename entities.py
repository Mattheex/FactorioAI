import numpy as np
import time

from var import *


class Case:
    def __init__(self, name, speed=0, limit=0, direction="all") -> None:
        self.items = []
        self.then = time.time()
        self.name = name
        self.speed = speed
        self.limit = limit
        self.direction = direction

    def itemNumber(self):
        return len(self.items)

    def itemsAvailable(self):
        return filter(lambda x: x.canMove(), self.items)

    def lenAvailables(self):
        l = 0
        for item in self.items:
            if item.canMove():
                l += 1
        return l

    def getItem(self):
        for item in self.items:
            if item.canMove():
                self.items.remove(item)
                item.move()
                return item

    def connectedToSupplier(self, nearby, center_pos, cls, canGet=None):
        if self.direction == "all":
            allSupplier = []
            for row in range(nearby.shape[0]):
                for col in range(nearby.shape[1]):
                    if (
                        (row != center_pos[0] or col != center_pos[1])
                        and row != col
                        and nearby[row, col].name == cls
                    ):
                        allSupplier.append(nearby[row, col])

            return allSupplier
        else:
            start = center_pos + canGet
            if nearby[start[0],start[1]].canSupply(start, center_pos):
                print(f"{nearby[*start]}")
                return nearby[*start].name == cls, nearby[*start]

        return False,False

    def action(self):
        duration_in_s = time.time() - self.then
        return duration_in_s * 1000 > self.speed

    def turn(self, nearby, center_pos):
        pass
    
    def canSupply(self, start, center_pos):
        return False


class Dirt(Case):
    def __init__(self) -> None:
        super().__init__("dirt")

    def __str__(self) -> str:
        return self.name


class Mine(Case):
    def __init__(self) -> None:
        super().__init__("mine", 500, 100)

    def canSupply(self, start, center_pos):
        return True

    def turn(self, nearby, center_pos):
        if self.action() and self.itemNumber() < self.limit:
            self.then = time.time()
            self.items.append(Item())

    def __str__(self) -> str:
        return f"{self.name} {self.itemNumber()}"


class Belt(Case):
    def __init__(self, directionStart, directionEnd) -> None:
        super().__init__("belt", 600, 2, directionStart)
        
        self.directionStart = directionStart
        self.directionEnd = directionEnd

        self.get = np.array([0, 0])
        self.give = np.array([0, 0])

        if directionStart == LEFT:
            self.get[0] = 0
            self.get[1] = -1
        elif directionStart == TOP:
            self.get[0] = -1
            self.get[1] = 0
        elif directionStart == DOWN:
            self.get[0] = 1
            self.get[1] = 0
        elif directionStart == RIGHT:
            self.get[0] = 0
            self.get[1] = 1
            
        if directionEnd == LEFT:
            self.give[0] = 0
            self.give[1] = -1
        elif directionEnd == TOP:
            self.give[0] = -1
            self.give[1] = 0
        elif directionEnd == DOWN:
            self.give[0] = 1
            self.give[1] = 0
        elif directionEnd == RIGHT:
            self.give[0] = 0
            self.give[1] = 1

    def canSupply(self, start, center_pos):
        return ((center_pos - start) == self.give).all()

    def turn(self, nearby, center_pos):
        if self.action():
            self.then = time.time()

            for cls in ["mine", "belt"]:
                isSupplier, supplier = self.connectedToSupplier(
                    nearby, center_pos, cls, self.get
                )

                print(isSupplier)
                if isSupplier:
                    print(f"{supplier} {supplier.lenAvailables()}")

                if (
                    isSupplier
                    and supplier.lenAvailables() > 0
                    and self.itemNumber() < self.limit
                ):
                    item = supplier.getItem()
                    self.items.append(item)

    def __str__(self) -> str:
        return f"{self.name} {self.itemNumber()} {self.directionStart[0]} -> {self.directionEnd[0]}"


class Chest(Case):
    def __init__(self) -> None:
        super().__init__("chest", 1000, 20)

    def canSupply(self, start, center_pos):
        return False

    def turn(self, nearby, center_pos):
        if self.action():
            self.then = time.time()

            allSupplier = self.connectedToSupplier(nearby, center_pos, "belt")

            for supplier in allSupplier:
                if supplier.lenAvailables() > 0 and self.itemNumber() < self.limit:
                    item = supplier.getItem()
                    self.items.append(item)

    def __str__(self) -> str:
        return f"{self.name} {self.itemNumber()}"


class Item:
    def __init__(self):
        self.speed = 1000
        self.then = time.time()

    def canMove(self):
        duration_in_s = time.time() - self.then
        print(duration_in_s)
        return duration_in_s * 1000 > self.speed

    def move(self):
        self.then = time.time()
