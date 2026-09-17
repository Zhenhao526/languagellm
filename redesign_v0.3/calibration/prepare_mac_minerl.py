"""Apply official MineRL changes and minimal local macOS build compatibility.

Run only after vanilla MCP-Reborn's `gradlew setup` has completed. This never
changes VPT. Environment calibration initialization is a separate explicit patch.
"""
from pathlib import Path
import difflib
import json
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE / 'vendor/minerl'
MCP = REPO / 'minerl/MCP-Reborn'
MARKER = BASE / 'logs/official_patch_applied.json'


def main():
    if not (MCP / 'src/main/java/net/minecraft/client/MainWindow.java').exists():
        raise RuntimeError('Vanilla setup has not produced mapped Minecraft sources')
    if not MARKER.exists():
        # Restore the two repository URL fixes used for the vanilla setup so
        # the published MineRL patch sees the exact upstream build file.
        shutil.copy2(MCP / 'build.gradle.upstream', MCP / 'build.gradle')
        if (MCP / 'README.md').exists():
            (MCP / 'README.md').rename(MCP / 'README.upstream.md')
        result = subprocess.run(['patch','--batch','-p','1','-i',
            str(REPO / 'scripts/mcp_patch.diff')],cwd=MCP,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (BASE / 'logs/official_minerl_patch.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError('Official patch failed; inspect log before retrying')
        shutil.copytree(REPO/'scripts/cursors',MCP/'src/main/resources/cursors',dirs_exist_ok=True)
        MARKER.write_text(json.dumps({'minerl_commit':subprocess.check_output(
            ['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()},indent=2))
    changed = []
    edits = {}
    gradle = MCP/'build.gradle'
    before = gradle.read_text()
    after = before.replace('jcenter()', 'maven { url = "https://maven.minecraftforge.net" }')
    after = after.replace("version: '3.2.1'", "version: '3.3.1'")
    after = after.replace('// compile group:', 'compile group:').replace('natives-osx', 'natives-macos')
    old_xjc = """        ant.xjc( destdir: 'src/main/java', package: 'com.microsoft.Malmo.Schemas' )
        {
            schema( dir: 'src/main/resources', includes: '*.xsd' )
        }"""
    new_xjc = """        // Old JAXB resolves relative includes incorrectly in non-ASCII paths.
        def localSchemas = java.nio.file.Files.createTempDirectory("language-minerl-schemas-").toFile()
        copy { from 'src/main/resources'; into localSchemas; include '*.xsd' }
        try {
            ant.xjc( destdir: 'src/main/java', package: 'com.microsoft.Malmo.Schemas' ) {
                schema( dir: localSchemas, includes: '*.xsd' )
            }
        } finally {
            delete localSchemas
        }"""
    after = after.replace(old_xjc, new_xjc)
    if '// Local macOS LWJGL version alignment' not in after:
        after += '''
// Local macOS LWJGL version alignment: no game or policy logic changes.
configurations.all {
    resolutionStrategy.eachDependency { DependencyResolveDetails details ->
        if (details.requested.group == 'org.lwjgl') {
            details.useVersion '3.3.1'
        }
    }
}
'''
    edits['build.gradle']=(before,after)
    launch=MCP/'launchClient.sh'
    before=launch.read_text()
    after=before.replace('java -Xmx$maxMem', 'java -XstartOnFirstThread -Xmx$maxMem')
    edits['launchClient.sh']=(before,after)
    window=MCP/'src/main/java/net/minecraft/client/MainWindow.java'
    before=window.read_text()
    after=before.replace('GLFW.glfwSetWindowIcon(this.handle, buffer);',
        '// macOS: custom GLFW window icons are unsupported; retain all other error checks.')
    edits[str(window.relative_to(MCP))]=(before,after)
    for name,(before,after) in edits.items():
        if before==after:
            continue
        (MCP/name).write_text(after)
        changed.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                        fromfile='official-minerl/'+name,tofile='local-macos/'+name))
    if changed:
        with (BASE/'macos_compatibility.patch').open('a') as stream:
            stream.write(''.join(changed))
    for name in ('gradlew','launchClient.sh'):
        p=MCP/name
        p.chmod(p.stat().st_mode|0o111)
    print(json.dumps({'official_patch':str(MARKER),'local_patch':str(BASE/'macos_compatibility.patch')}))


if __name__=='__main__':
    main()
