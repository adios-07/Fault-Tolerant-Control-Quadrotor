import os
import shutil
import glob
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory, get_package_prefix


def launch_setup(context, *args, **kwargs):
    world_name = os.path.splitext(
        LaunchConfiguration('world_name').perform(context))[0]
    drone_model = LaunchConfiguration('drone_model').perform(context)
    px4_dir = os.path.expanduser(LaunchConfiguration('px4_dir').perform(context))
    headless = LaunchConfiguration('headless').perform(context) == 'true'
    rtf = float(LaunchConfiguration('rtf').perform(context))

    pkg_share_dir = get_package_share_directory('rotor_fail')
    worlds_dir = os.path.join(pkg_share_dir, 'worlds')
    models_dir = os.path.join(pkg_share_dir, 'models')
    install_dir = os.path.join(pkg_share_dir, '..')

    build_dir = os.path.join(px4_dir, 'build', 'px4_sitl_default')
    rootfs_dir = os.path.join(build_dir, 'rootfs')
    px4_binary = os.path.join(build_dir, 'bin', 'px4')
    romfs_dir = os.path.join(px4_dir, 'ROMFS', 'px4fmu_common')
    rcs_script = os.path.join(romfs_dir, 'init.d-posix', 'rcS')

    # Sandbox copy of the environment; never mutate os.environ
    env = os.environ.copy()

    # ---- Brain: airframe (independent of body) ----
    # PX4_ROMFS_EXTRA no longer exists in this PX4 (v1.18 main).
    # Equivalent: stage the airframe into the generated build rootfs (not the source tree).
    env['PX4_SYS_AUTOSTART'] = '9000'
    airframes_rootfs = os.path.join(build_dir, 'etc', 'init.d-posix', 'airframes')
    airframe_src = os.path.join(pkg_share_dir, 'ext_romfs', 'init.d-posix', 'airframes',
                                '9000_custom_x500')
    airframe_dst = os.path.join(airframes_rootfs, '9000_custom_x500')
    os.makedirs(airframes_rootfs, exist_ok=True)
    shutil.copy2(airframe_src, airframe_dst)
    print(f"[INFO] Staged custom airframe -> {airframe_dst}")

    # Clear cached parameters so airframe edits apply automatically
    eeprom_dir = os.path.join(rootfs_dir, 'eeprom')
    if os.path.exists(eeprom_dir):
        for p in glob.glob(os.path.join(eeprom_dir, 'parameters_*')):
            os.remove(p)
            print(f"[INFO] Deleted cached PX4 parameters: {p}")

    # ---- Body: Gazebo model (independent of brain) ----
    env['PX4_SIM_MODEL'] = f'gz_{drone_model}'  # -> spawns model named '<drone_model>'
    env['PX4_GZ_MODELS'] = models_dir           # out-of-tree model directory
    env['PX4_GZ_WORLDS'] = worlds_dir           # out-of-tree world directory
    env['PX4_GZ_WORLD'] = world_name
    env['PX4_SIM_LOCKSTEP'] = '1'

    # ---- Gazebo resource paths ----
    px4_gz_models = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'models')
    px4_gz_worlds = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'worlds')
    env['GZ_SIM_RESOURCE_PATH'] = (
        f'{install_dir}:{models_dir}:{px4_gz_models}:{px4_gz_worlds}')
    env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = ':'.join([
        os.path.join(build_dir, 'src', 'modules', 'simulation', 'gz_plugins'),
        os.path.join(get_package_prefix('rotor_fail'), 'lib')])  # RotorFaultPlugin
    env['GZ_IP'] = '127.0.0.1'

    world_sdf = os.path.join(worlds_dir, world_name + '.sdf')

    actions = [
        # World server FIRST: px4-rc.gzsim skips launching its own world (and skips
        # sourcing gz_env.sh, which would override our PX4_GZ_MODELS/PX4_GZ_WORLDS).
        ExecuteProcess(
            # -z: physics update rate; the world steps 1 ms, so rtf 2 = 2000 Hz (PX4 lockstep follows)
            cmd=['gz', 'sim', '-r', '-s', '-v', '4', world_sdf] + (['-z', str(1000 * rtf)] if rtf != 1.0 else []),
            cwd=rootfs_dir,
            env=env,
            output='screen'
        ),
        # Raw PX4 binary: no make, no sitl_run.sh. cwd=build_dir so ${R}etc/...
        # resolves to the staged ROMFS and gz_env.sh is NOT sourced.
        ExecuteProcess(
            cmd=[px4_binary, '-s', rcs_script],
            cwd=build_dir,
            env=env,
            output='screen'
        ),
    ]
    if not headless:
        actions.append(ExecuteProcess(cmd=['gz', 'sim', '-g'], cwd=rootfs_dir, env=env, output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world_name',
            default_value='empty',
            description='World file in worlds/ (without .sdf)'
        ),
        DeclareLaunchArgument(
            'drone_model',
            default_value='custom_x500',
            description='Model dir in models/ (body only, decoupled from airframe)'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='true = no Gazebo GUI (benchmark runs)'
        ),
        DeclareLaunchArgument(
            'rtf',
            default_value='1.0',
            description='real-time factor: > 1 runs the sim faster than real time (headless benchmarks)'
        ),
        DeclareLaunchArgument(
            'px4_dir',
            default_value='~/PX4-Autopilot',
            description='Path to the PX4-Autopilot source directory'
        ),
        OpaqueFunction(function=launch_setup),
    ])