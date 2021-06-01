"""A wrapper for engaging with the THOR environment."""

import copy
import functools
import math
import random
import typing
import warnings
from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping

import ai2thor.server
import networkx as nx
import numpy as np
from ai2thor.controller import Controller
import os

VISIBILITY_DISTANCE = 1.5
DEFAULT_FOV = 90.0


def round_to_factor(num: float, base: int) -> int:
    """Rounds floating point number to the nearest integer multiple of the
    given base. E.g., for floating number 90.1 and integer base 45, the result
    is 90.

    # Attributes

    num : floating point number to be rounded.
    base: integer base
    """
    return round(num / base) * base


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position['x'], cam_position['z'])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        # if len(position) == 3:
        #     x, _, z = position
        # else:
        #     x, z = position

        camera_position = (np.array((position['x'], position['z'])) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def safe_display(frame):
    """ Plots frame in a safe way, dealing with matplotlib"""
    os.environ['DISPLAY'] = 'magellanic:10.0'
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("tkagg", warn=False)
    plt.imshow(frame)
    plt.show()
    # os.environ['DISPLAY'] = ':0.0'

class AI2ThorEnvironment(object):
    """Wrapper for the ai2thor controller providing additional functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
     documentation on AI2-THOR.

    # Attributes

    controller : The ai2thor controller.
    """

    def __init__(
            self,
            x_display: Optional[str] = None,
            docker_enabled: bool = False,
            local_thor_build: Optional[str] = None,
            visibility_distance: float = VISIBILITY_DISTANCE,
            fov: float = DEFAULT_FOV,
            player_screen_width: int = 300,
            player_screen_height: int = 300,
            quality: str = "Very Low",
            restrict_to_initially_reachable_points: bool = False,
            make_agents_visible: bool = True,
            object_open_speed: float = 1.0,
            simplify_physics: bool = False,
            render_depth_image=False,
            render_class_image=False,
            render_object_image=False,
    ) -> None:
        """Initializer.

        # Parameters

        x_display : The x display into which to launch ai2thor (possibly necessarily if you are running on a server
            without an attached display).
        docker_enabled : Whether or not to run thor in a docker container (useful on a server without an attached
            display so that you don't have to start an x display).
        local_thor_build : The path to a local build of ai2thor. This is probably not necessary for your use case
            and can be safely ignored.
        visibility_distance : The distance (in meters) at which objects, in the viewport of the agent,
            are considered visible by ai2thor and will have their "visible" flag be set to `True` in the metadata.
        fov : The agent's camera's field of view.
        player_screen_width : The width resolution (in pixels) of the images returned by ai2thor.
        player_screen_height : The height resolution (in pixels) of the images returned by ai2thor.
        quality : The quality at which to render. Possible quality settings can be found in
            `ai2thor._quality_settings.QUALITY_SETTINGS`.
        restrict_to_initially_reachable_points : Whether or not to restrict the agent to locations in ai2thor
            that were found to be (initially) reachable by the agent (i.e. reachable by the agent after resetting
            the scene). This can be useful if you want to ensure there are only a fixed set of locations where the
            agent can go.
        make_agents_visible : Whether or not the agent should be visible. Most noticable when there are multiple agents
            or when quality settings are high so that the agent casts a shadow.
        object_open_speed : How quickly objects should be opened. High speeds mean faster simulation but also mean
            that opening objects have a lot of kinetic energy and can, possibly, knock other objects away.
        simplify_physics : Whether or not to simplify physics when applicable. Currently this only simplies object
            interactions when opening drawers (when simplified, objects within a drawer do not slide around on
            their own when the drawer is opened or closed, instead they are effectively glued down).
        """

        self._start_player_screen_width = player_screen_width
        self._start_player_screen_height = player_screen_height
        self._local_thor_build = local_thor_build
        self.x_display = x_display
        self.controller: Optional[Controller] = None
        self._started = False
        self._quality = quality

        self._initially_reachable_points: Optional[List[Dict]] = None
        self._initially_reachable_points_set: Optional[Set[Tuple[float, float]]] = None
        self._move_mag: Optional[float] = None
        self._grid_size: Optional[float] = None
        self._visibility_distance = visibility_distance
        self._fov = fov
        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.make_agents_visible = make_agents_visible
        self.object_open_speed = object_open_speed
        self._always_return_visible_range = False
        self.simplify_physics = simplify_physics


        self.render_depth_image = render_depth_image
        self.render_class_image = render_class_image
        self.render_object_image = render_object_image

        self.start(None)
        self.controller.docker_enabled = docker_enabled  # type: ignore

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.controller.last_event.frame

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    @property
    def started(self) -> bool:
        """Has the ai2thor controller been started."""
        return self._started

    @property
    def last_action(self) -> str:
        """Last action, as a string, taken by the agent."""
        return self.controller.last_event.metadata["lastAction"]

    @last_action.setter
    def last_action(self, value: str) -> None:
        """Set the last action taken by the agent.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["lastAction"] = value

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @last_action_success.setter
    def last_action_success(self, value: bool) -> None:
        """Set whether or not the last action taken by the agent was a success.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["lastActionSuccess"] = value

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.controller.last_event.metadata["actionReturn"]

    @last_action_return.setter
    def last_action_return(self, value: Any) -> None:
        """Set the value returned by the last action.

        Doing this is rewriting history, be careful.
        """
        self.controller.last_event.metadata["actionReturn"] = value

    def start(
            self, scene_name: Optional[str], move_mag: float = 0.25, **kwargs,
    ) -> None:
        """Starts the ai2thor controller if it was previously stopped.

        After starting, `reset` will be called with the scene name and move magnitude.

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to reset.
        """
        if self._started:
            raise RuntimeError(
                "Trying to start the environment but it is already started."
            )

        def create_controller():
            return Controller(
                x_display=self.x_display,
                player_screen_width=self._start_player_screen_width,
                player_screen_height=self._start_player_screen_height,
                local_executable_path=self._local_thor_build,
                quality=self._quality,
            )

        self.controller = create_controller()

        if (
                self._start_player_screen_height,
                self._start_player_screen_width,
        ) != self.current_frame.shape[:2]:
            self.controller.step(
                {
                    "action": "ChangeResolution",
                    "x": self._start_player_screen_width,
                    "y": self._start_player_screen_height,
                }
            )

        self._started = True
        self.reset(scene_name=scene_name, move_mag=move_mag, **kwargs)

    def stop(self) -> None:
        """Stops the ai2thor controller."""
        try:
            self.controller.stop()
        except Exception as e:
            warnings.warn(str(e))
        finally:
            self._started = False

    def randomize(self, force_visible=False):
        random_seed = random.randint(0, 2**32)
        self.___last_seed = random_seed
        self.controller.step(action='InitialRandomSpawn', randomSeed=random_seed,
                             forceVisible=force_visible, numPlacementAttempts=3,
                        placeStationary=True)
        # TODO: find a replacement for....
        # self.controller.step(action='RandomToggleStateOfAllObjects', randomSeed=random_seed)

        # PROBLEM: Sometimes there are objects that are too close to us. Kill them
        # for agent_num in range(self.num_agents):
        # agent_pos = self.get_agent_location()
        # for obj in self.all_objects():
        #     x_dist = np.abs(agent_pos['x'] - obj['position']['x'])
        #     z_dist = np.abs(agent_pos['z'] - obj['position']['z'])
        #     dist = max(x_dist, z_dist)
        #     if dist < 0.25:
        #         self.controller.step(action='RemoveFromScene', objectId=obj['objectId'])

        # Get initially reachable positions again?
        self.recompute_reachable()

    def reset(self, scene_name: Optional[str], move_mag: float = 0.25, **kwargs):
        """Resets the ai2thor in a new scene.

        Resets ai2thor into a new scene and initializes the scene/agents with
        prespecified settings (e.g. move magnitude).

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to the controller "Initialize" action.
        """
        self._move_mag = move_mag
        self._grid_size = self._move_mag

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]
        self.controller.reset(scene_name)

        self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self._grid_size,
                "visibilityDistance": self._visibility_distance,
                "fieldOfView": self._fov,
                "makeAgentsVisible": self.make_agents_visible,
                "alwaysReturnVisibleRange": self._always_return_visible_range,
                # 'agentCount': self.num_agents,
                'renderClassImage': self.render_class_image,
                'renderObjectImage': self.render_object_image,
                'renderDepthImage': self.render_depth_image,
                **kwargs,
            }
        )

        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )
        self.recompute_reachable()


    def recompute_reachable(self):
        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return


    def teleport_agent_to(
            self,
            x: float,
            y: float,
            z: float,
            rotation: float,
            horizon: float,
            standing: Optional[bool] = None,
            force_action: bool = False,
            only_initially_reachable: Optional[bool] = None,
            verbose=True,
            ignore_y_diffs=False,
    ) -> None:
        """Helper function teleporting the agent to a given location."""
        if standing is None:
            standing = self.last_event.metadata['agent']["isStanding"]
        original_location = self.get_agent_location()
        target = {"x": x, "y": y, "z": z}
        if only_initially_reachable is None:
            only_initially_reachable = self.restrict_to_initially_reachable_points
        if only_initially_reachable:
            reachable_points = self.initially_reachable_points
            reachable = False
            for p in reachable_points:
                if self.position_dist(target, p, ignore_y=ignore_y_diffs) < 0.01:
                    reachable = True
                    break
            if not reachable:
                self.last_action = "TeleportFull"
                self.last_event.metadata[
                    "errorMessage"
                ] = "Target position was not initially reachable."
                self.last_action_success = False
                return
        self.controller.step(
            dict(
                action="TeleportFull",
                x=x,
                y=y,
                z=z,
                rotation={"x": 0.0, "y": rotation, "z": 0.0},
                horizon=horizon,
                standing=standing,
                forceAction=force_action, # agentId=agent_num,
            )
        )
        if not self.last_action_success:
            agent_location = self.get_agent_location()
            rot_diff = (
                               agent_location["rotation"] - original_location["rotation"]
                       ) % 360
            new_old_dist = self.position_dist(
                original_location, agent_location, ignore_y=ignore_y_diffs
            )
            if (
                    self.position_dist(
                        original_location, agent_location, ignore_y=ignore_y_diffs
                    )
                    > 1e-2
                    or min(rot_diff, 360 - rot_diff) > 1
            ):
                warnings.warn(
                    "Teleportation FAILED but agent still moved (position_dist {}, rot diff {})"
                    " (\nprevious location\n{}\ncurrent_location\n{}\n)".format(
                        new_old_dist, rot_diff, original_location, agent_location
                    )
                )
            return

        if force_action:
            assert self.last_action_success
            return

        agent_location = self.get_agent_location()
        rot_diff = (agent_location["rotation"] - rotation) % 360
        if (
                self.position_dist(agent_location, target, ignore_y=ignore_y_diffs) > 1e-2
                or min(rot_diff, 360 - rot_diff) > 1
        ):
            if only_initially_reachable:
                self._snap_agent_to_initially_reachable(verbose=False)
            if verbose:
                warnings.warn(
                    "Teleportation did not place agent"
                    " precisely where desired in scene {}"
                    " (\ndesired\n{}\nactual\n{}\n)"
                    " perhaps due to grid snapping."
                    " Action is considered failed but agent may have moved.".format(
                        self.scene_name,
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "rotation": rotation,
                            "standing": standing,
                            "horizon": horizon,
                        },
                        agent_location,
                    )
                )
                p2 = {'x': x, 'y': y, 'z': z}
                for o in self.all_objects():
                    if self.position_dist(o['position'], p2) < 0.5:
                        warnings.warn("Close object {}? {} to {}".format(o['objectId'], o['position'], p2))
                # warnings.warn([o['position'] for o in self.all_objects() if self.position_dist(o['position'], p2) < 0.25])
            self.last_action_success = False
        return

    def random_reachable_state(self) -> Dict:
        """Returns a random reachable location in the scene."""
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice([0, 90, 180, 270])
        horizon = random.choice([0, 30, 60, 330])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(self, partial_position: Optional[Dict[str, float]] = None) -> Dict:
        """Teleports the agent to a random reachable location in the scene."""
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            state = self.random_reachable_state()
            self.teleport_agent_to(**{**state, **partial_position})
            k += 1
            if k % 2 == 0:
                self.randomize(force_visible=True)

        if not self.last_action_success:
            warnings.warn(
                (
                    "Randomize agent location in scene {}"
                    " and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, partial_position)
            )
            self.teleport_agent_to(**{**state, **partial_position}, force_action=True)  # type: ignore
            assert self.last_action_success

        assert state is not None
        return state

    def object_pixels_in_frame(
            self, object_id: str, hide_all: bool = True, hide_transparent: bool = False
    ) -> np.ndarray:
        """Return an mask for a given object in the agent's current view.

        # Parameters

        object_id : The id of the object.
        hide_all : Whether or not to hide all other objects in the scene before getting the mask.
        hide_transparent : Whether or not partially transparent objects are considered to occlude the object.

        # Returns

        A numpy array of the mask.
        """

        # Emphasizing an object turns it magenta and hides all other objects
        # from view, we can find where the hand object is on the screen by
        # emphasizing it and then scanning across the image for the magenta pixels.
        if hide_all:
            self.step({"action": "EmphasizeObject", "objectId": object_id})
        else:
            self.step({"action": "MaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "HideTranslucentObjects"})
        # noinspection PyShadowingBuiltins
        filter = np.array([[[255, 0, 255]]])
        object_pixels = 1 * np.all(self.current_frame == filter, axis=2)
        if hide_all:
            self.step({"action": "UnemphasizeAll"})
        else:
            self.step({"action": "UnmaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "UnhideAllObjects"})
        return object_pixels

    def object_pixels_on_grid(
            self,
            object_id: str,
            grid_shape: Tuple[int, int],
            hide_all: bool = True,
            hide_transparent: bool = False,
    ) -> np.ndarray:
        """Like `object_pixels_in_frame` but counts object pixels in a
        partitioning of the image."""

        def partition(n, num_parts):
            m = n // num_parts
            parts = [m] * num_parts
            num_extra = n % num_parts
            for k in range(num_extra):
                parts[k] += 1
            return parts

        object_pixels = self.object_pixels_in_frame(
            object_id=object_id, hide_all=hide_all, hide_transparent=hide_transparent
        )

        # Divide the current frame into a grid and count the number
        # of hand object pixels in each of the grid squares
        sums_in_blocks: List[List] = []
        frame_shape = self.current_frame.shape[:2]
        row_inds = np.cumsum([0] + partition(frame_shape[0], grid_shape[0]))
        col_inds = np.cumsum([0] + partition(frame_shape[1], grid_shape[1]))
        for i in range(len(row_inds) - 1):
            sums_in_blocks.append([])
            for j in range(len(col_inds) - 1):
                sums_in_blocks[i].append(
                    np.sum(
                        object_pixels[
                        row_inds[i]: row_inds[i + 1], col_inds[j]: col_inds[j + 1]
                        ]
                    )
                )
        return np.array(sums_in_blocks, dtype=np.float32)

    def object_in_hand(self):
        """Object metadata for the object in the agent's hand."""
        inv_objs = self.last_event.metadata["inventoryObjects"]
        if len(inv_objs) == 0:
            return None
        elif len(inv_objs) == 1:
            return self.get_object_by_id(
                self.last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        else:
            raise AttributeError("Must be <= 1 inventory objects.")

    @property
    def initially_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that were
        reachable after initially resetting."""
        assert self._initially_reachable_points is not None
        return copy.deepcopy(self._initially_reachable_points)  # type:ignore

    @property
    def initially_reachable_points_set(self) -> Set[Tuple[float, float]]:
        """Set of (x,z) locations in the scene that were reachable after
        initially resetting."""
        if self._initially_reachable_points_set is None:
            self._initially_reachable_points_set = set()
            for p in self.initially_reachable_points:
                self._initially_reachable_points_set.add(
                    self._agent_location_to_tuple(p)
                )

        return self._initially_reachable_points_set

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.step({"action": "GetReachablePositions"})
        return self.last_event.metadata["reachablePositions"]  # type:ignore

    def get_agent_location(self) -> Dict[str, Union[float, bool]]:
        """Gets agent's location."""
        metadata = self.controller.last_event.metadata
        location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata['agent']["isStanding"],
        }
        return location

    @staticmethod
    def _agent_location_to_tuple(p: Dict[str, float]) -> Tuple[float, float]:
        return (round(p["x"], 2), round(p["z"], 2))

    def _snap_agent_to_initially_reachable(self, verbose=True):
        agent_location = self.get_agent_location()

        end_location_tuple = self._agent_location_to_tuple(agent_location)
        if end_location_tuple in self.initially_reachable_points_set:
            return

        agent_x = agent_location["x"]
        agent_z = agent_location["z"]

        closest_reachable_points = list(self.initially_reachable_points_set)
        closest_reachable_points = sorted(
            closest_reachable_points,
            key=lambda xz: abs(xz[0] - agent_x) + abs(xz[1] - agent_z),
        )

        # In rare cases end_location_tuple might be not considered to be in self.initially_reachable_points_set
        # even when it is, here we check for such cases.
        if (
                math.sqrt(
                    (
                            (
                                    np.array(closest_reachable_points[0])
                                    - np.array(end_location_tuple)
                            )
                            ** 2
                    ).sum()
                )
                < 1e-6
        ):
            return

        saved_last_action = self.last_action
        saved_last_action_success = self.last_action_success
        saved_last_action_return = self.last_action_return
        saved_error_message = self.last_event.metadata["errorMessage"]

        # Thor behaves weirdly when the agent gets off of the grid and you
        # try to teleport the agent back to the closest grid location. To
        # get around this we first teleport the agent to random location
        # and then back to where it should be.
        for point in self.initially_reachable_points:
            if abs(agent_x - point["x"]) > 0.1 or abs(agent_z - point["z"]) > 0.1:
                self.teleport_agent_to(
                    rotation=0,
                    horizon=30,
                    **point,
                    only_initially_reachable=False,
                    verbose=False,
                )
                if self.last_action_success:
                    break

        for p in closest_reachable_points:
            self.teleport_agent_to(
                **{**agent_location, "x": p[0], "z": p[1]},
                only_initially_reachable=False,
                verbose=False,
            )
            if self.last_action_success:
                break

        teleport_forced = False
        if not self.last_action_success:
            self.teleport_agent_to(
                **{
                    **agent_location,
                    "x": closest_reachable_points[0][0],
                    "z": closest_reachable_points[0][1],
                },
                force_action=True,
                only_initially_reachable=False,
                verbose=False,
            )
            teleport_forced = True

        self.last_action = saved_last_action
        self.last_action_success = saved_last_action_success
        self.last_action_return = saved_last_action_return
        self.last_event.metadata["errorMessage"] = saved_error_message
        new_agent_location = self.get_agent_location()
        if verbose:
            warnings.warn(
                (
                    "In {}, at location (x,z)=({},{}) which is not in the set "
                    "of initially reachable points;"
                    " attempting to correct this: agent teleported to (x,z)=({},{}).\n"
                    "Teleportation {} forced."
                ).format(
                    self.scene_name,
                    agent_x,
                    agent_z,
                    new_agent_location["x"],
                    new_agent_location["z"],
                    "was" if teleport_forced else "wasn't",
                )
            )

    def step(
            self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        action = typing.cast(str, action_dict["action"])

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        if self.simplify_physics:
            action_dict["simplifyOPhysics"] = True

        if ("Move" in action and "Hand" not in action) or ('JumpAhead' in action):  # type: ignore
            action_dict = {
                **action_dict,
                "moveMagnitude": self._move_mag,
            }  # type: ignore
            if action == 'JumpAhead':
                action_dict['action'] = 'MoveAhead'
                action_dict['moveMagnitude'] = 4 * self._move_mag

            start_location = self.get_agent_location()
            sr = self.controller.step(action_dict)

            if '_start_state_key' in action_dict:
                if not self.get_key(start_location) == action_dict['_start_state_key']:
                    import ipdb
                    ipdb.set_trace()
            if '_end_state_key' in action_dict:
                if not self.get_key(self.get_agent_location()) == action_dict['_end_state_key']:
                    if not self.get_key(self.get_agent_location()) == action_dict['_start_state_key']:
                        raise RuntimeError("we're not keeping track of the state keys correctly")

            # Double check
            if self.restrict_to_initially_reachable_points:
                end_location_tuple = self._agent_location_to_tuple(
                    self.get_agent_location()
                )
                if end_location_tuple not in self.initially_reachable_points_set:
                    self.teleport_agent_to(**start_location, force_action=True)  # type: ignore
                    self.last_action = action
                    self.last_action_success = False
                    self.last_event.metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
        elif "RandomizeHideSeekObjects" in action:
            last_position = self.get_agent_location()
            self.controller.step(action_dict)
            metadata = self.last_event.metadata
            if self.position_dist(last_position, self.get_agent_location()) > 0.001:
                self.teleport_agent_to(**last_position, force_action=True)  # type: ignore
                warnings.warn(
                    "In scene {}, after randomization of hide and seek objects, agent moved.".format(
                        self.scene_name
                    )
                )

            sr = self.controller.step({"action": "GetReachablePositions"})
            self._initially_reachable_points = self.controller.last_event.metadata[
                "reachablePositions"
            ]
            self._initially_reachable_points_set = None
            self.last_action = action
            self.last_action_success = metadata["lastActionSuccess"]
            self.controller.last_event.metadata["reachablePositions"] = []
        elif "RotateUniverse" in action:
            sr = self.controller.step(action_dict)
            metadata = self.last_event.metadata

            if metadata["lastActionSuccess"]:
                sr = self.controller.step({"action": "GetReachablePositions"})
                self._initially_reachable_points = self.controller.last_event.metadata[
                    "reachablePositions"
                ]
                self._initially_reachable_points_set = None
                self.last_action = action
                self.last_action_success = metadata["lastActionSuccess"]
                self.controller.last_event.metadata["reachablePositions"] = []
        else:
            sr = self.controller.step(action_dict)

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr

    @staticmethod
    def position_dist(
            p0: Mapping[str, Any], p1: Mapping[str, Any], ignore_y: bool = False
    ) -> float:
        """Distance between two points of the form {"x": x, "y":y, "z":z"}."""
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )

    @staticmethod
    def rotation_dist(a: Dict[str, float], b: Dict[str, float]):
        """Distance between rotations."""

        def deg_dist(d0: float, d1: float):
            dist = (d0 - d1) % 360
            return min(dist, 360 - dist)

        return sum(deg_dist(a[k], b[k]) for k in ["x", "y", "z"])

    def closest_object_with_properties(
            self, properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that has the given
        properties."""
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        min_dist = float("inf")
        closest = None
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                d = self.position_dist(agent_pos, o["position"])
                if d < min_dist:
                    min_dist = d
                    closest = o
        return closest

    def closest_visible_object_of_type(
            self, object_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that is visible and has the
        given type."""
        properties = {"visible": True, "objectType": object_type}
        return self.closest_object_with_properties(properties)

    def closest_object_of_type(self, object_type: str) -> Optional[Dict[str, Any]]:
        """Find the object closest to the agent that has the given type."""
        properties = {"objectType": object_type}
        return self.closest_object_with_properties(properties)

    def closest_reachable_point_to_position(
            self, position: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """Of all reachable positions, find the one that is closest to the
        given location."""
        target = np.array([position["x"], position["z"]])
        min_dist = float("inf")
        closest_point = None
        for pt in self.initially_reachable_points:
            dist = np.linalg.norm(target - np.array([pt["x"], pt["z"]]))
            if dist < min_dist:
                closest_point = pt
                min_dist = dist
                if min_dist < 1e-3:
                    break
        assert closest_point is not None
        return closest_point, min_dist

    @staticmethod
    def _angle_from_to(a_from: float, a_to: float) -> float:
        a_from = a_from % 360
        a_to = a_to % 360
        min_rot = min(a_from, a_to)
        max_rot = max(a_from, a_to)
        rot_across_0 = (360 - max_rot) + min_rot
        rot_not_across_0 = max_rot - min_rot
        rot_err = min(rot_across_0, rot_not_across_0)
        if rot_across_0 == rot_err:
            rot_err *= -1 if a_to > a_from else 1
        else:
            rot_err *= 1 if a_to > a_from else -1
        return rot_err

    def agent_xz_to_scene_xz(self, agent_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()

        x_rel_agent = agent_xz["x"]
        z_rel_agent = agent_xz["z"]
        scene_x = agent_pos["x"]
        scene_z = agent_pos["z"]
        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            scene_x += x_rel_agent
            scene_z += z_rel_agent
        elif abs(rotation - 90) < 1e-5:
            scene_x += z_rel_agent
            scene_z += -x_rel_agent
        elif abs(rotation - 180) < 1e-5:
            scene_x += -x_rel_agent
            scene_z += -z_rel_agent
        elif abs(rotation - 270) < 1e-5:
            scene_x += -z_rel_agent
            scene_z += x_rel_agent
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": scene_x, "z": scene_z}

    def scene_xz_to_agent_xz(self, scene_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()
        x_err = scene_xz["x"] - agent_pos["x"]
        z_err = scene_xz["z"] - agent_pos["z"]

        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            agent_x = x_err
            agent_z = z_err
        elif abs(rotation - 90) < 1e-5:
            agent_x = -z_err
            agent_z = x_err
        elif abs(rotation - 180) < 1e-5:
            agent_x = -x_err
            agent_z = -z_err
        elif abs(rotation - 270) < 1e-5:
            agent_x = z_err
            agent_z = -x_err
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": agent_x, "z": agent_z}

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
            self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})

    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                return o
        return None

    ###
    # Following is used for computing shortest paths between states
    ###
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}

    GRAPH_ACTIONS_SET = {"RotateLeft", "RotateRight", "MoveAhead"}

    def reachable_points_with_rotations_and_horizons(self):
        self.controller.step({"action": "GetReachablePositions"})
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in [0, 90, 180, 270]:
            for p in points_slim:
                p = copy.copy(p)
                p["rotation"] = r
                points.append(p)
        return points

    @staticmethod
    def location_for_key(key, y_value=0.0):
        x, z, rot = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot)
        return loc

    @staticmethod
    def get_key(input_dict: Dict[str, Any]) -> Tuple[float, float, int]:
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            rot = input_dict["rotation"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            rot = input_dict["rotation"]["y"]

        return (
            round(x, 2),
            round(z, 2),
            round_to_factor(rot, 90) % 360,
        )

    def update_graph_with_failed_action(self, failed_action: str):
        if (
                self.scene_name not in self._CACHED_GRAPHS
                or failed_action not in self.GRAPH_ACTIONS_SET
        ):
            return

        source_key = self.get_key(self.last_event.metadata["agent"])
        edge_dict = self.graph[source_key]
        to_remove_key = None
        for target_key in self.graph[source_key]:
            if edge_dict[target_key]["action"] == failed_action:
                to_remove_key = target_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(
            self,
            g: nx.DiGraph,
            s: Tuple[float, float, int],
            t: Tuple[float, float, int],
    ):
        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot = s
        t_x, t_z, t_rot = t

        dist = round(math.sqrt((s_x - t_x) ** 2 + (s_z - t_z) ** 2), 2)
        angle_dist = (round_to_factor(t_rot - s_rot, 90) % 360) // 90

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [dist, angle_dist]) != 1:
            return

        grid_size = self._grid_size
        action = None
        if angle_dist != 0:
            if angle_dist == 1:
                action = "RotateRight"
            elif angle_dist == 3:
                action = "RotateLeft"

        elif ae(dist, grid_size):
            if (
                    (s_rot == 0 and ae(t_z - s_z, grid_size))
                    or (s_rot == 90 and ae(t_x - s_x, grid_size))
                    or (s_rot == 180 and ae(t_z - s_z, -grid_size))
                    or (s_rot == 270 and ae(t_x - s_x, -grid_size))
            ):
                g.add_edge(s, t, action="MoveAhead")

        if action is not None:
            g.add_edge(s, t, action=action)

    @functools.lru_cache(1)
    def possible_neighbor_offsets(self) -> Tuple[Tuple[float, float, int], ...]:
        grid_size = round(self._grid_size, 2)
        offsets = []
        for rot_diff in [-90, 0, 90]:
            for x_diff in [-grid_size, 0, grid_size]:
                for z_diff in [-grid_size, 0, grid_size]:
                    if (rot_diff != 0) + (x_diff != 0) + (z_diff != 0) == 1:
                        offsets.append((x_diff, z_diff, rot_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: Tuple[float, float, int]):
        if s in graph:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for o in self.possible_neighbor_offsets():
            t = (s[0] + o[0], s[1] + o[1], (s[2] + o[2]) % 360)
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    @property
    def graph(self):
        if self.scene_name not in self._CACHED_GRAPHS:
            g = nx.DiGraph()
            points = self.reachable_points_with_rotations_and_horizons()
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._CACHED_GRAPHS[self.scene_name] = g
        return self._CACHED_GRAPHS[self.scene_name]

    @graph.setter
    def graph(self, g):
        self._CACHED_GRAPHS[self.scene_name] = g

    def _check_contains_key(self, key: Tuple[float, float, int], add_if_not=True):
        if key not in self.graph:
            warnings.warn(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)

    def shortest_state_path(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        try:
            path = nx.shortest_path(self.graph, source_state_key, goal_state_key)
            return path
        except Exception as _:
            return None

    def action_transitioning_between_keys(self, s, t):
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        if source_state_key == goal_state_key:
            raise RuntimeError("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)

        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_path_length(self, source_state_key, goal_state_key):
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")

    @staticmethod
    def _horizon_transition(start_horizon: int=0, end_horizon: int=0):
        """
        Uses LookDown a bunch to switch horizons
        :param start_horizon:
        :param end_horizon:
        :return:
        """
        delta = round_to_factor(end_horizon - start_horizon, 30) % 360
        if delta > 180:
            delta -= 360

        delta_steps = delta // 30
        res = []
        for i in range(abs(delta_steps)):
            res.append({
                'action': 'LookDown' if delta_steps > 0 else 'LookUp',
                '_start_horizon': round_to_factor(start_horizon + 30 * i * delta_steps / abs(delta_steps), 30),
                '_end_horizon': round_to_factor(start_horizon + 30 * (i+1) * delta_steps / abs(delta_steps), 30),
            })
        return res

    @staticmethod
    def _fix_multi_moves(res_list):
        """
        Given a list with keys 'action' we'll replace multiple moves with JumpAhead (4 moves)
        starting at the front
        :param res_list:
        :return:
        """
        new_res = []
        i = 0
        while i < len(res_list):
            item = res_list[i]
            if item['action'] != 'MoveAhead':
                new_res.append(item)
                i += 1
                continue
            same_action = np.array([x['action'] == item['action'] for x in res_list[i:]], dtype=np.bool)
            num_sequential = np.cumprod(same_action).sum()

            if num_sequential >= 4:
                new_res.append({
                    'action': 'JumpAhead',
                    '_start_state_key': item['_start_state_key'],
                    '_end_state_key': res_list[i+3]['_end_state_key']
                })
                i += 4
            else:
                new_res.append(item)
                i += 1
        return new_res

    @staticmethod
    def _fix_strafe(res_list, num_inner_max=2):
        """
        Replace R / F F / L with a strafe
        :param res_list:
        :return:
        """
        action_map = {'MoveAhead': 'f', 'RotateRight': 'r', 'RotateLeft': 'l'}

        action_sequence_map = {}
        for i in range(num_inner_max):
            action_sequence_map['r{}l'.format('f' * (i+1))] = ('MoveRight', i+1)
            action_sequence_map['l{}r'.format('f' * (i+1))] = ('MoveLeft', i+1)

        new_res = []
        i = 0
        while i < len(res_list):
            item = res_list[i]
            start_angle = item['_start_state_key'][2]

            # Detect patterns
            action_plan = ''.join([action_map[x['action']] for x in res_list[i:]])
            if action_plan.startswith(('rrr', 'lll')):
                import ipdb
                ipdb.set_trace()

            new_plan = None
            for j in range(num_inner_max):
                if action_plan[:(3+j)] in action_sequence_map:
                   new_plan = action_sequence_map[action_plan[:3+j]]

            # No strafe plan
            if new_plan is None:
                i += 1
                new_res.append(item)
            else:
                for j in range(new_plan[1]):
                    new_res.append({
                        'action': new_plan[0],
                        '_start_state_key': item['_start_state_key'] if j == 0 else new_res[-1]['_end_state_key'],
                        '_end_state_key': (res_list[i + 1 + j]['_end_state_key'][0],
                                           res_list[i + 1 + j]['_end_state_key'][1],
                                           start_angle,)
                    })
                i += new_plan[1] + 2
        return new_res

    def get_fancy_shortest_path(self, start_location, end_location, fix_multi_moves=True, num_inner_max=2):
        """
        Gets shortest path with a bunch of bells and whistles:

        * If we're moving -- we'll set horizon position to be reasonable
        * Replace right/forward/left with a strafe

        :param start_location:
        :param end_location:
        :return: Sequence of actions
        """

        # First get the shortest path, ignoring horizons
        start_key = self.get_key(start_location)
        end_key = self.get_key(end_location)
        sp = self.shortest_state_path(start_key, end_key)

        if sp is None:
            return None

        res = []
        for (k1, k2) in zip(sp[:-1], sp[1:]):
            action = self.graph.get_edge_data(k1, k2)['action']
            # print(env.graph.get_edge_data(k1, k2)['action'])

            res.append({
                'action': action,
                '_start_state_key': k1,
                '_end_state_key': k2,
            })
        if len(res) == 0:
            return res
        # import json
        # print("ORIGINAL PLAN: {}".format(json.dumps(res, indent=2)), flush=True)
        res = self._fix_strafe(res, num_inner_max=num_inner_max)
        if fix_multi_moves:
            res = self._fix_multi_moves(res)

        # last - add horizons
        if len(res) > 3:
            res = self._horizon_transition(start_horizon=start_location['horizon']) + res + self._horizon_transition(
                end_horizon=end_location['horizon'])
        else:
            res += self._horizon_transition(start_horizon=start_location['horizon'], end_horizon=end_location['horizon'])
        return res






