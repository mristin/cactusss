"""Steer the balloon with your head through a thorny cactus desert."""
import abc
import argparse
import enum
import fractions
import importlib.resources
import inspect
import math
import os.path
import pathlib
import random
import sys
import time
from typing import (
    Optional,
    List,
    Union,
    Tuple,
    Sequence,
    Final,
    cast,
    Iterator,
    Set,
    Type,
)

import cv2
import pygame
import pygame.freetype
from icontract import require, ensure, DBC

import cactusss
import cactusss.events
import cactusss.heading
from cactusss.common import assert_never

assert cactusss.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)


class Media:
    """Represent all the media loaded in the main memory from the file system."""

    def __init__(
        self,
        ground_tiles: List[pygame.surface.Surface],
        balloon_sprites: List[pygame.surface.Surface],
        plopping_sprites: List[pygame.surface.Surface],
        cactus_sprites: List[pygame.surface.Surface],
        benign_noise_sprites: List[pygame.surface.Surface],
        droplet_sprite: pygame.surface.Surface,
        sun_sprite: pygame.surface.Surface,
        wind_sprite: pygame.surface.Surface,
        font: pygame.freetype.Font,  # type: ignore
        plop_sound: pygame.mixer.Sound,
        happy_end_sound: pygame.mixer.Sound,
    ) -> None:
        """Initialize with the given values."""
        self.ground_tiles = ground_tiles
        self.balloon_sprites = balloon_sprites
        self.plopping_sprites = plopping_sprites
        self.cactus_sprites = cactus_sprites
        self.benign_noise_sprites = benign_noise_sprites
        self.droplet_sprite = droplet_sprite
        self.sun_sprite = sun_sprite
        self.wind_sprite = wind_sprite
        self.font = font
        self.plop_sound = plop_sound
        self.happy_end_sound = happy_end_sound


SCENE_WIDTH = 640
SCENE_HEIGHT = 320

TILE_WIDTH = 32
TILE_HEIGHT = 32
ROAD_WIDTH_IN_TILES = 14
ROAD_LEFT_MARGIN = (SCENE_WIDTH - ROAD_WIDTH_IN_TILES * TILE_WIDTH) / 2

#: Forward velocity of the balloon relative to the ground, in world coordinates
BALLOON_FORWARD_VELOCITY = 50

#: Sidewards velocity of the balloon relative to the ground, in world coordinates
BALLOON_SIDEWARDS_VELOCITY = 60

NORMAL_BALLOON_WIDTH = 24
NORMAL_BALLOON_HEIGHT = 48

#: In seconds
PLOP_DURATION = 0.5

#: Roughly the length of the level in tiles.
#:
#: The actual level length might be a bit above that number.
EXPECTED_LEVEL_LENGTH_IN_TILES = 100
assert EXPECTED_LEVEL_LENGTH_IN_TILES >= SCENE_HEIGHT / TILE_HEIGHT


@ensure(lambda result: (result[0] is not None) ^ (result[1] is not None))
def load_media() -> Tuple[Optional[Media], Optional[str]]:
    """Load the media from the file system."""
    images_dir = PACKAGE_DIR / "media/images"

    ground_tile_pths = sorted(images_dir.glob("ground*.png"))
    ground_tiles = [
        pygame.image.load(str(pth)).convert_alpha() for pth in ground_tile_pths
    ]
    if len(ground_tiles) == 0:
        return None, f"No 'ground*.png' found in images directory {images_dir}"

    for i, ground_tile in enumerate(ground_tiles):
        if ground_tile.get_width() != 32 or ground_tile.get_height() != 32:
            return None, (
                f"Expected the tile {ground_tile_pths[i]} to be 32x32, "
                f"got {ground_tile.get_width()}x{ground_tile.get_height()}"
            )

    balloon_sprite_pths = sorted(images_dir.glob("balloon*.png"))
    balloon_sprites = [
        pygame.image.load(str(pth)).convert_alpha() for pth in balloon_sprite_pths
    ]
    if len(balloon_sprites) == 0:
        return None, f"No 'balloon*.png' found in images directory {images_dir}"

    for i, balloon_sprite in enumerate(balloon_sprites):
        if (
            balloon_sprite.get_width() != NORMAL_BALLOON_WIDTH
            or balloon_sprite.get_height() != NORMAL_BALLOON_HEIGHT
        ):
            return None, (
                f"Expected the tile {balloon_sprite_pths[i]} to be "
                f"{NORMAL_BALLOON_WIDTH}x{NORMAL_BALLOON_HEIGHT}, "
                f"got {balloon_sprite.get_width()}x{balloon_sprite.get_height()}"
            )

    plopping_sprite_pths = sorted(images_dir.glob("plopping*.png"))
    plopping_sprites = [
        pygame.image.load(str(pth)).convert_alpha() for pth in plopping_sprite_pths
    ]
    if len(plopping_sprites) == 0:
        return None, f"No 'plopping*.png' found in images directory {images_dir}"

    for i, plopping_sprite in enumerate(plopping_sprites):
        if plopping_sprite.get_width() != 32 or plopping_sprite.get_height() != 64:
            return None, (
                f"Expected the tile {plopping_sprite_pths[i]} to be 32x64, "
                f"got {plopping_sprite.get_width()}x{plopping_sprite.get_height()}"
            )

    cactus_sprite_pths = sorted(images_dir.glob("cactus*.png"))
    cactus_sprites = [
        pygame.image.load(str(pth)).convert_alpha() for pth in cactus_sprite_pths
    ]
    if len(cactus_sprites) == 0:
        return None, f"No 'cactus*.png' found in images directory {images_dir}"

    benign_noise_sprite_pths = sorted(images_dir.glob("benign_noise*.png"))
    benign_noise_sprites = [
        pygame.image.load(str(pth)).convert_alpha() for pth in benign_noise_sprite_pths
    ]
    if len(benign_noise_sprites) == 0:
        return None, f"No 'benign_noise*.png' found in images directory {images_dir}"

    for i, benign_noise_sprite in enumerate(benign_noise_sprites):
        if (
            benign_noise_sprite.get_width() != 32
            or benign_noise_sprite.get_height() != 32
        ):
            return None, (
                f"Expected the tile {benign_noise_sprite_pths[i]} to be 32x32, "
                f"got {benign_noise_sprite.get_width()}x{benign_noise_sprite.get_height()}"
            )

    droplet_sprite_pth = images_dir / "droplet.png"
    if not droplet_sprite_pth.exists():
        return None, f"Wind sprite not found: {droplet_sprite_pth}"

    droplet_sprite = pygame.image.load(str(images_dir / "droplet.png")).convert_alpha()
    if droplet_sprite.get_width() != 32 or droplet_sprite.get_height() != 32:
        return None, (
            f"Expected the sprite {droplet_sprite_pth} to be 32x32, "
            f"got {droplet_sprite.get_width()}x{droplet_sprite.get_height()}"
        )

    sun_sprite_pth = images_dir / "sun.png"
    if not sun_sprite_pth.exists():
        return None, f"Wind sprite not found: {sun_sprite_pth}"

    sun_sprite = pygame.image.load(str(images_dir / "sun.png")).convert_alpha()
    if sun_sprite.get_width() != 32 or sun_sprite.get_height() != 32:
        return None, (
            f"Expected the sprite {sun_sprite_pth} to be 32x32, "
            f"got {sun_sprite.get_width()}x{sun_sprite.get_height()}"
        )

    wind_sprite_pth = images_dir / "wind.png"
    if not wind_sprite_pth.exists():
        return None, f"Wind sprite not found: {wind_sprite_pth}"

    wind_sprite = pygame.image.load(str(images_dir / "wind.png")).convert_alpha()
    if wind_sprite.get_width() != 32 or wind_sprite.get_height() != 32:
        return None, (
            f"Expected the sprite {wind_sprite_pth} to be 32x32, "
            f"got {wind_sprite.get_width()}x{wind_sprite.get_height()}"
        )

    return (
        Media(
            ground_tiles=ground_tiles,
            balloon_sprites=balloon_sprites,
            plopping_sprites=plopping_sprites,
            cactus_sprites=cactus_sprites,
            benign_noise_sprites=benign_noise_sprites,
            droplet_sprite=droplet_sprite,
            sun_sprite=sun_sprite,
            wind_sprite=wind_sprite,
            # fmt: off
        font=pygame.freetype.Font(  # type: ignore
            str(PACKAGE_DIR / "media/fonts/freesansbold.ttf")
        ),
            # fmt: on
            plop_sound=pygame.mixer.Sound(str(PACKAGE_DIR / "media/sfx/plop.ogg")),
            happy_end_sound=pygame.mixer.Sound(
                str(PACKAGE_DIR / "media/sfx/happy_end.ogg")
            ),
        ),
        None,
    )


class PassableTile:
    """Represent a tile which does not plop the balloon."""

    #: Appearance of the tile
    sprite: Final[Optional[pygame.surface.Surface]]

    #: Specify the mod to be applied if picked on this tile
    mod_factory: Optional[Type["Mod"]]

    def __init__(
        self,
        sprite: Optional[pygame.surface.Surface],
        mod_factory: Optional[Type["Mod"]] = None,
    ) -> None:
        """Initialize with the given values."""
        self.sprite = sprite
        self.mod_factory = mod_factory


class CactusTile:
    """Represent an obstacle that plops the balloon if touched."""

    sprite: Final[pygame.surface.Surface]

    def __init__(self, sprite: pygame.surface.Surface) -> None:
        """Initialize with the given values."""
        self.sprite = sprite


Tile = Union[PassableTile, CactusTile]


class Level:
    """Represent different layers of the game level."""

    #: Ground tiles, as (row, column).
    #:
    #: Row 0 starts at the *bottom* of the screen.
    ground: Final[Sequence[Sequence[pygame.surface.Surface]]]

    #: Define the paths through the level, as (row, column).
    #:
    #: Row 0 starts at the *bottom* of the screen.
    tiles: Final[Sequence[Sequence[Tile]]]

    @require(
        lambda ground, tiles: len(ground) == len(tiles)
        and all(
            len(ground_row) == len(tiles_row) == ROAD_WIDTH_IN_TILES
            for ground_row, tiles_row in zip(ground, tiles)
        ),
        "Ground and tiles have same dimensions and cover the road width",
    )
    def __init__(
        self,
        ground: Sequence[Sequence[pygame.surface.Surface]],
        tiles: Sequence[Sequence[Tile]],
    ) -> None:
        """Initialize with the given values."""
        self.ground = ground
        self.tiles = tiles


class LevelPattern(Sequence[str]):
    """Represent a pre-defined level pattern."""

    # fmt: off
    @require(
        lambda data:
        all(
            all(
                symbol in ('#', ' ')
                for symbol in row
            )
            for row in data
        ),
        "Only expected symbols"
    )
    @require(
        lambda data:
        all(
            len(row) == ROAD_WIDTH_IN_TILES
            for row in data
        ),
        "Whole width of the road covered"
    )
    # fmt: on
    def __new__(cls, data: Sequence[str]) -> "LevelPattern":
        return cast(LevelPattern, data)

    def __getitem__(self, index: int) -> str:
        raise NotImplementedError("Only for type annotation.")

    def __len__(self) -> int:
        raise NotImplementedError("Only for type annotation.")

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError("Only for type annotation.")


#: Patterns to generate the road. "#" means obstacle, " " means empty.
PATTERNS = [
    LevelPattern(
        [
            "####       ###",
            "####       ###",
            "####       ###",
        ]
    ),
    LevelPattern(
        [
            "####       ###",
            "####       ###",
            "######     ###",
            "####       ###",
            "####       ###",
        ]
    ),
    LevelPattern(
        [
            "###        ###",
            "##    #   ####",
            "#    #    ####",
            "#   #    #####",
            "#   #    #####",
            "#    #   #####",
            "##        ####",
        ]
    ),
    LevelPattern(
        [
            "####        ##",
            "####    #   ##",
            "####    #    #",
            "####    #    #",
            "####     #   #",
            "####    #    #",
            "###         ##",
        ]
    ),
    LevelPattern(
        [
            "###        ###",
            "##     #     #",
            "##     #    ##",
            "#     ##    ##",
            "##     #     #",
            "###         ##",
        ]
    ),
    LevelPattern(
        [
            "##         ###",
            "#     #      #",
            "#    ##      #",
            "#     ##     #",
            "#     #      #",
            "##          ##",
        ]
    ),
]


@require(lambda expected_level_length: expected_level_length > 0)
def generate_level(
    media: Media,
    expected_level_length: int = EXPECTED_LEVEL_LENGTH_IN_TILES,
    patterns: Optional[Sequence[LevelPattern]] = None,
) -> Level:
    """
    Generate randomly a level.

    We pick the patterns randomly, so the actual level length might slightly exceed
    the ``expected_level_length``.
    """
    patterns = PATTERNS if patterns is None else patterns

    # We start with the first pattern so that the start is an easy one.
    symbol_matrix = [row for row in patterns[0]]  # type: List[str]

    while len(symbol_matrix) < expected_level_length:
        pattern = random.choice(patterns)
        for row in pattern:
            symbol_matrix.append(row)

    level_as_pattern = LevelPattern(symbol_matrix)

    ground = []  # type: List[List[pygame.surface.Surface]]
    tiles = []  # type: List[List[Tile]]

    for row in level_as_pattern:
        ground.append(
            [random.choice(media.ground_tiles) for _ in range(ROAD_WIDTH_IN_TILES)]
        )

        tile_row = []  # type: List[Tile]
        for symbol in row:
            if symbol == " ":
                if random.random() < 0.5:
                    sprite = None
                else:
                    sprite = random.choice(media.benign_noise_sprites)

                tile_row.append(PassableTile(sprite=sprite))
            elif symbol == "#":
                tile_row.append(CactusTile(sprite=random.choice(media.cactus_sprites)))

        tiles.append(tile_row)

    next_mod = 5 + random.randint(1, 5)
    while next_mod < len(tiles):
        tile_row = tiles[next_mod]
        potential_passables = [
            tile for tile in tile_row if isinstance(tile, PassableTile)
        ]  # type: List[PassableTile]

        tile_with_mod = random.choice(potential_passables)
        tile_with_mod.mod_factory = random.choice(MOD_FACTORIES)

        next_mod = next_mod + random.randint(1, 3)

    return Level(ground=ground, tiles=tiles)


class Mod(DBC):
    """Modify some balloon property over time."""

    #: Time stamp when the mod has been picked
    start: Final[float]

    #: Time stamp when the mod should not apply anymore
    eta: Final[float]

    def __init__(self, start: float) -> None:
        self.start = start
        self.eta = start + 5.0

    @abc.abstractmethod
    def apply(self, balloon: "Balloon") -> None:
        """Apply the mod on the balloon state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def undo(self, balloon: "Balloon") -> None:
        """Restore the balloon state back to the state before the mod."""
        raise NotImplementedError()


class DropletMod(Mod):
    """Shrink the balloon."""

    def apply(self, balloon: "Balloon") -> None:
        balloon.scale = 0.7
        balloon.scale_mod = self

    def undo(self, balloon: "Balloon") -> None:
        balloon.scale = 1.0
        balloon.scale_mod = None


class SunMod(Mod):
    """Grow the balloon."""

    def apply(self, balloon: "Balloon") -> None:
        balloon.scale = 1.3
        balloon.scale_mod = self

    def undo(self, balloon: "Balloon") -> None:
        balloon.scale = 1.0
        balloon.scale_mod = None


ScaleMod = Union[DropletMod, SunMod]


class WindMod(Mod):
    """Accelerate the balloon in the forward direction."""

    def apply(self, balloon: "Balloon") -> None:
        balloon.velocity = (balloon.velocity[0], BALLOON_FORWARD_VELOCITY + 50)
        balloon.wind_mod = self

    def undo(self, balloon: "Balloon") -> None:
        balloon.velocity = (balloon.velocity[0], BALLOON_FORWARD_VELOCITY)
        balloon.wind_mod = None


MOD_FACTORIES = (DropletMod, SunMod, WindMod)


def _assert_mod_factories_exhaustive() -> None:
    """Assert that we did not forget any mod in the :py:attr:`MOD_FACTORIES`."""
    got = set(cls.__name__ for cls in MOD_FACTORIES)

    expected = set(
        name
        for name, symbol in globals().items()
        if isinstance(symbol, type) and issubclass(symbol, Mod) and symbol is not Mod
    )

    if got != expected:
        raise AssertionError(
            f"Unexpected MOD_FACTORIES: got {sorted(got)}, expected {sorted(expected)}"
        )


_assert_mod_factories_exhaustive()


class Balloon:
    """Represent the state of the balloon."""

    #: World position, in the world coordinates as (x, y).
    #:
    #: Y-axis starts at the bottom of the world (as opposed to the screen coordinates)
    xy: Tuple[float, float]

    #: Velocity in the world coordinates, as (x, y).
    #:
    #: Y-axis starts at the bottom of the world (as opposed to the screen coordinates)
    velocity: Tuple[float, float]

    #: Scale with the respect to the normal balloon size
    scale: float

    #: Mod that influences the size of the balloon
    scale_mod: Optional[ScaleMod]

    #: Mod that influences the velocity of the balloon
    wind_mod: Optional[WindMod]

    def __init__(self, xy: Tuple[float, float], velocity: Tuple[float, float]) -> None:
        """Initialize the balloon state with the given values."""
        self.xy = xy
        self.velocity = velocity
        self.scale = 1.0
        self.scale_mod = None
        self.wind_mod = None

    def width(self) -> int:
        """Width of the balloon, in pixels."""
        return round(self.scale * NORMAL_BALLOON_WIDTH)

    def height(self) -> int:
        """Height of the balloon, in pixels."""
        return round(self.scale * NORMAL_BALLOON_HEIGHT)


class Command(enum.Enum):
    """Represent a command given to the balloon."""

    GO_LEFT = 0
    GO_RIGHT = 1


class GameOver:
    #: Timestamp when the game ended
    end: Final[float]

    kind: Final[cactusss.events.GameOverKind]

    def __init__(self, end: float, kind: cactusss.events.GameOverKind) -> None:
        """Initialize with the given values."""
        self.end = end
        self.kind = kind


class State:
    """Capture the global state of the game."""

    #: Set if we received the signal to quit the game
    received_quit: bool

    #: Timestamp when the game started, seconds since epoch
    game_start: float

    #: Current clock in the game, seconds since epoch
    now: float

    #: Set when the game finishes
    game_over: Optional[GameOver]

    level: Level

    balloon: Balloon

    active_commands: Set[Command]

    def __init__(self, game_start: float, level: Level) -> None:
        """Initialize with the given values and the defaults."""
        initialize_state(self, game_start, level)


def initialize_state(state: State, game_start: float, level: Level) -> None:
    """Initialize the state to the start one."""
    state.received_quit = False
    state.game_start = game_start
    state.now = game_start
    state.game_over = None

    state.level = level
    state.balloon = Balloon(
        xy=(ROAD_WIDTH_IN_TILES * TILE_WIDTH / 2, NORMAL_BALLOON_HEIGHT),
        velocity=(0, BALLOON_FORWARD_VELOCITY),
    )
    state.active_commands = set()


@require(lambda xmin_a, xmax_a: xmin_a <= xmax_a)
@require(lambda ymin_a, ymax_a: ymin_a <= ymax_a)
@require(lambda xmin_b, xmax_b: xmin_b <= xmax_b)
@require(lambda ymin_b, ymax_b: ymin_b <= ymax_b)
def intersect(
    xmin_a: Union[int, float],
    ymin_a: Union[int, float],
    xmax_a: Union[int, float],
    ymax_a: Union[int, float],
    xmin_b: Union[int, float],
    ymin_b: Union[int, float],
    xmax_b: Union[int, float],
    ymax_b: Union[int, float],
) -> bool:
    """Return true if the two bounding boxes intersect."""
    return (xmin_a <= xmax_b and xmax_a >= xmin_b) and (
        ymin_a <= ymax_b and ymax_a >= ymin_b
    )


def handle_in_game(
    state: State, our_event_queue: List[cactusss.events.EventUnion], media: Media
) -> None:
    """Consume the first action in the queue during the game."""
    if len(our_event_queue) == 0:
        return

    event = our_event_queue.pop(0)

    now = pygame.time.get_ticks() / 1000

    if isinstance(event, cactusss.events.Tick):
        time_delta = now - state.now
        state.now = now

        if (
            Command.GO_LEFT not in state.active_commands
            and Command.GO_RIGHT not in state.active_commands
        ) or (
            Command.GO_LEFT in state.active_commands
            and Command.GO_RIGHT in state.active_commands
        ):
            state.balloon.velocity = (0.0, state.balloon.velocity[1])
        elif Command.GO_LEFT in state.active_commands:
            state.balloon.velocity = (
                -BALLOON_SIDEWARDS_VELOCITY,
                state.balloon.velocity[1],
            )
        elif Command.GO_RIGHT in state.active_commands:
            state.balloon.velocity = (
                BALLOON_SIDEWARDS_VELOCITY,
                state.balloon.velocity[1],
            )
        else:
            pass

        # Undo mods?
        for maybe_mod in (state.balloon.scale_mod, state.balloon.wind_mod):
            if maybe_mod is not None and now > maybe_mod.eta:
                maybe_mod.undo(state.balloon)

        state.balloon.xy = (
            max(
                0.0,
                min(
                    state.balloon.xy[0] + time_delta * state.balloon.velocity[0],
                    ROAD_WIDTH_IN_TILES * TILE_WIDTH - state.balloon.width(),
                ),
            ),
            state.balloon.xy[1] + time_delta * state.balloon.velocity[1],
        )

        # region Balloon tile

        # The world y-axis starts at the bottom, not at the top. We check
        # at the center of the balloon if we have a collision to simulate a faux
        # 3D scene.
        row = int((state.balloon.xy[1] - state.balloon.height() / 2) / TILE_HEIGHT)

        column = int(state.balloon.xy[0] / TILE_WIDTH)

        # NOTE (mristin, 2023-01-16):
        # We have to account for the position of the balloon relative to the tiles
        # as the balloon is narrower than a tile.
        balloon_rest_in_tile = state.balloon.xy[0] - column * TILE_WIDTH
        if balloon_rest_in_tile > TILE_WIDTH - state.balloon.width() / 2:
            column += 1

        # endregion

        assert 0 <= row, f"{row=}, {column=}, {state.balloon.xy=}"

        # Finish line?
        if row >= len(state.level.tiles):
            our_event_queue.append(
                cactusss.events.GameOver(kind=cactusss.events.GameOverKind.HAPPY_END)
            )
            return

        assert (
            0 <= column < len(state.level.tiles[row])
        ), f"{row=}, {column=}, {state.balloon.xy=}"
        tile = state.level.tiles[row][column]

        # Plop?
        if isinstance(tile, CactusTile):
            our_event_queue.append(
                cactusss.events.GameOver(kind=cactusss.events.GameOverKind.PLOP)
            )
            return

        # Pick up mod?
        if isinstance(tile, PassableTile) and tile.mod_factory is not None:
            mod = tile.mod_factory(start=now)
            mod.apply(state.balloon)
            tile.mod_factory = None

    else:
        # Ignore the event
        pass


def handle(
    state: State, our_event_queue: List[cactusss.events.EventUnion], media: Media
) -> None:
    """Consume the first action in the queue."""
    if len(our_event_queue) == 0:
        return

    if isinstance(our_event_queue[0], cactusss.events.ReceivedQuit):
        our_event_queue.pop(0)
        state.received_quit = True

    elif isinstance(our_event_queue[0], cactusss.events.ReceivedRestart):
        our_event_queue.pop(0)
        pygame.mixer.stop()
        initialize_state(
            state,
            game_start=pygame.time.get_ticks() / 1000,
            level=generate_level(media=media),
        )

    elif isinstance(our_event_queue[0], cactusss.events.GameOver):
        event = our_event_queue[0]
        our_event_queue.pop(0)

        if state.game_over is None:
            state.game_over = GameOver(end=state.now, kind=event.kind)

            if state.game_over.kind is cactusss.events.GameOverKind.HAPPY_END:
                media.happy_end_sound.play()
            elif state.game_over.kind is cactusss.events.GameOverKind.PLOP:
                media.plop_sound.play()
            else:
                assert_never(state.game_over)
    else:
        handle_in_game(state, our_event_queue, media)


def render_game_over(state: State, media: Media) -> pygame.surface.Surface:
    """Render the "game over" dialogue as a scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((128, 128, 128))

    assert state.game_over is not None

    if state.game_over.kind is cactusss.events.GameOverKind.HAPPY_END:
        road_length = len(state.level.tiles) * TILE_HEIGHT
        time_delta = state.game_over.end - state.game_start
        average_velocity = road_length / time_delta
        media.font.render_to(scene, (20, 20), "You made it!", (255, 255, 255), size=16)

        media.font.render_to(
            scene,
            (20, 60),
            f"Average velocity: {average_velocity:.1f} pixels / second",
            (255, 255, 255),
            size=16,
        )
    elif state.game_over.kind is cactusss.events.GameOverKind.PLOP:
        media.font.render_to(scene, (20, 20), "Game Over :'(", (255, 255, 255), size=16)

        sprite_i = int((state.now / 0.5)) % (len(media.plopping_sprites) + 1)
        if sprite_i < len(media.plopping_sprites):
            sprite = media.plopping_sprites[sprite_i]
            scene.blit(sprite, (SCENE_WIDTH / 2 - sprite.get_width() / 2, 40))
    else:
        assert_never(state.game_over.kind)

    media.font.render_to(
        scene,
        (20, 300),
        'Press "q" to quit and "r" to restart',
        (255, 255, 255),
        size=10,
    )

    return scene


def render_quit(media: Media) -> pygame.surface.Surface:
    """Render the "Quitting..." dialogue as a scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))
    scene.fill((0, 0, 0))

    media.font.render_to(scene, (20, 20), "Quitting...", (255, 255, 255), size=32)

    return scene


def world_xy_to_screen_xy(
    xy: Tuple[float, float],
    camera_start: float,
) -> Tuple[float, float]:
    """
    Convert the world coordinates to screen coordinates, as (x, y).

    Camera_start denotes the bottom of the camera view in the world coordinates.

    World y-axis starts at the bottom of the world, while the screen y-axis starts
    at the top of the screen.

    Screen covers the whole scene.
    """
    return (ROAD_LEFT_MARGIN + xy[0], SCENE_HEIGHT - (xy[1] - camera_start))


def tile_row_column_to_screen_xy(
    row: int, column: int, camera_start: float
) -> Tuple[float, float]:
    """
    Convert the tile world coordinates (as row, column) to screen coordinates as (x, y).

    Camera_start denotes the bottom of the camera view in the world coordinates.

    Tile world starts at the bottom of the world, while the screen y-axis starts
    at the top of the screen.

    Screen covers the whole scene.
    """
    return (
        ROAD_LEFT_MARGIN + column * TILE_WIDTH,
        SCENE_HEIGHT - (row * TILE_HEIGHT + TILE_HEIGHT - camera_start),
    )


def cvmat_to_surface(image: cv2.Mat) -> pygame.surface.Surface:
    """Convert from OpenCV to pygame."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    return pygame.image.frombuffer(image_rgb.tobytes(), (width, height), "RGB")


def render_game(
    state: State, media: Media, recognizer: cactusss.heading.Recognizer
) -> pygame.surface.Surface:
    """Render the game scene."""
    scene = pygame.surface.Surface((SCENE_WIDTH, SCENE_HEIGHT))

    max_camera_start = len(state.level.ground) * TILE_HEIGHT - SCENE_HEIGHT

    camera_start = min(
        int(state.balloon.xy[1] - state.balloon.height()), max_camera_start
    )

    assert camera_start >= 0

    tile_i_start = int(camera_start / TILE_HEIGHT)

    # Exclusive
    tile_i_end = min(
        tile_i_start + int(SCENE_HEIGHT / TILE_HEIGHT) + 1, len(state.level.ground)
    )

    for ground_i in range(tile_i_start, tile_i_end):
        ground_row = state.level.ground[ground_i]

        for ground_j, ground_sprite in enumerate(ground_row):
            screen_xy = tile_row_column_to_screen_xy(
                row=ground_i, column=ground_j, camera_start=camera_start
            )

            scene.blit(ground_sprite, screen_xy)

    for tile_i in range(tile_i_start, tile_i_end):
        tile_row = state.level.tiles[tile_i]

        for tile_j, tile in enumerate(tile_row):
            screen_xy = tile_row_column_to_screen_xy(
                row=tile_i, column=tile_j, camera_start=camera_start
            )

            if tile.sprite is not None:
                scene.blit(tile.sprite, screen_xy)

            if isinstance(tile, PassableTile) and tile.mod_factory is not None:
                mod_sprite = None  # type: Optional[pygame.surface.Surface]
                if tile.mod_factory is DropletMod:
                    mod_sprite = media.droplet_sprite
                elif tile.mod_factory is SunMod:
                    mod_sprite = media.sun_sprite
                elif tile.mod_factory is WindMod:
                    mod_sprite = media.wind_sprite
                else:
                    assert_never(tile.mod_factory)

                assert mod_sprite is not None
                scene.blit(mod_sprite, screen_xy)

    screen_xy = world_xy_to_screen_xy(state.balloon.xy, camera_start=camera_start)

    balloon_sprite_i = int(state.now / 0.5) % len(media.balloon_sprites)
    scaled_balloon_sprite = pygame.transform.scale(
        media.balloon_sprites[balloon_sprite_i],
        (state.balloon.width(), state.balloon.height()),
    )

    scene.blit(scaled_balloon_sprite, screen_xy)

    media.font.render_to(
        scene,
        (ROAD_LEFT_MARGIN + 10, 10),
        'Press "q" to quit and "r" to restart',
        (0, 0, 0),
        size=12,
    )

    if recognizer.head_image is not None:
        head = cvmat_to_surface(recognizer.head_image)

        head_canvas = pygame.surface.Surface((50, 50))
        resize_image_to_canvas_and_blit(head, head_canvas)

        scene.blit(head_canvas, (scene.get_width() - head_canvas.get_width(), 0))

    return scene


def render(
    state: State, media: Media, recognizer: cactusss.heading.Recognizer
) -> pygame.surface.Surface:
    """Render the state of the program."""
    if state.received_quit:
        return render_quit(media)

    if state.game_over is not None:
        return render_game_over(state, media)

    return render_game(state, media, recognizer)


def resize_image_to_canvas_and_blit(
    image: pygame.surface.Surface, canvas: pygame.surface.Surface
) -> None:
    """Draw the image on canvas resizing it to maximum at constant aspect ratio."""
    canvas.fill((0, 0, 0))

    canvas_aspect_ratio = fractions.Fraction(canvas.get_width(), canvas.get_height())
    image_aspect_ratio = fractions.Fraction(image.get_width(), image.get_height())

    if image_aspect_ratio < canvas_aspect_ratio:
        new_image_height = canvas.get_height()
        new_image_width = image.get_width() * (new_image_height / image.get_height())

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_width() - image.get_width()) / 2)

        canvas.blit(image, (margin, 0))

    elif image_aspect_ratio == canvas_aspect_ratio:
        new_image_width = canvas.get_width()
        new_image_height = image.get_height()

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        canvas.blit(image, (0, 0))
    else:
        new_image_width = canvas.get_width()
        new_image_height = int(
            image.get_height() * (new_image_width / image.get_width())
        )

        image = pygame.transform.scale(image, (new_image_width, new_image_height))

        margin = int((canvas.get_height() - image.get_height()) / 2)

        canvas.blit(image, (0, margin))


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )

    # NOTE (mristin, 2022-12-16):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(cactusss.__version__)
        return 0

    args = parser.parse_args()

    pygame.init()
    pygame.mixer.pre_init()
    pygame.mixer.init()

    pygame.display.set_caption("Cactusss")

    surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    print("Loading the media...")
    try:
        media, error = load_media()

        if error is not None:
            print(f"Failed to load the media: {error}", file=sys.stderr)
            return 1

        assert media is not None
    except Exception as exception:
        print(
            f"Failed to load the media: {exception.__class__.__name__} {exception}",
            file=sys.stderr,
        )
        return 1

    now = pygame.time.get_ticks() / 1000
    clock = pygame.time.Clock()

    print("Initializing the state...")
    state = State(game_start=now, level=generate_level(media=media))

    our_event_queue = []  # type: List[cactusss.events.EventUnion]

    # Reuse the tick object so that we don't have to create it every time
    tick_event = cactusss.events.Tick()

    print("Entering the endless loop...")

    recognizer = cactusss.heading.Recognizer()
    recognizer.start()

    print("Waiting for capture to be opened...")
    wait_start = time.time()
    while not recognizer.capture_is_opened:
        delta = time.time() - wait_start
        if delta > 5:
            print(
                "Could not open the video capture in more than 5 seconds. "
                "Something went wrong.",
                file=sys.stderr,
            )
            return 1

    try:
        while not state.received_quit and recognizer.capture_is_opened:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    our_event_queue.append(cactusss.events.ReceivedQuit())

                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    our_event_queue.append(cactusss.events.ReceivedQuit())

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    our_event_queue.append(cactusss.events.ReceivedRestart())

                else:
                    # Ignore the event that we do not handle
                    pass

            # region Active commands

            active_commands = set()

            # NOTE (mristin, 2023-01-15):
            # We add keys as alternative to key presses for people who simply want to
            # demo the game without the webcam, but we don't document this explicitly to
            # motivate the users to relax their neck (and not make it more stiff).
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                active_commands.add(Command.GO_LEFT)

            if keys[pygame.K_RIGHT]:
                active_commands.add(Command.GO_RIGHT)

            if cactusss.heading.Pose.LEFT in recognizer.active_poses:
                active_commands.add(Command.GO_LEFT)

            if cactusss.heading.Pose.RIGHT in recognizer.active_poses:
                active_commands.add(Command.GO_RIGHT)

            state.active_commands = active_commands

            # endregion

            our_event_queue.append(tick_event)

            while len(our_event_queue) > 0:
                handle(state, our_event_queue, media)

            scene = render(state, media, recognizer)
            resize_image_to_canvas_and_blit(scene, surface)
            pygame.display.flip()

            # Enforce 30 frames per second
            clock.tick(30)
    finally:
        print("Quitting the game...")
        tic = time.time()
        recognizer.close()
        pygame.quit()
        print(f"Quit the game after: {time.time() - tic:.2f} seconds")

    return 0


def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="cactusss")


if __name__ == "__main__":
    sys.exit(main(prog="cactusss"))
