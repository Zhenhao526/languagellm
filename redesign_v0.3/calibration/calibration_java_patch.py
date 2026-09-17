#!/usr/bin/env python3
"""Add bounded, native Minecraft initialization to generated MineRL EnvServer.java.

Does not edit vendor sources unless explicitly invoked with --source and --apply.
The added path applies only to Mission Summary values beginning ``Calib_``.
No policy code, action mapping, observations, or resource mechanics are replaced.

API checks: the vendored MineRL 1.0 scripts/mcp_patch.diff layout;
MCP 1.16.5 SRG bytecode (javap) and Forge snapshot 20201028-1.16.3 mappings.
The generated source still requires javac and a real reset test before use.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

MARKER = "// CALIBRATION_NATIVE_INIT_V1"

IMPORTS = """
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.entity.ai.attributes.Attributes;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.network.play.server.SHeldItemChangePacket;
import net.minecraft.network.play.server.SUpdateHealthPacket;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.server.ServerWorld;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import javax.xml.parsers.DocumentBuilderFactory;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
"""

METHODS = r'''
    // CALIBRATION_NATIVE_INIT_V1
    // Receipt is evaluator metadata, never an alternative image or policy input.
    private volatile JsonObject calibrationInitReceipt = null;

    private static Element calibrationElement(Element root, String name) {
        NodeList matches = root.getElementsByTagNameNS("*", name);
        if (matches.getLength() != 1) {
            throw new IllegalArgumentException("Calibration requires one " + name);
        }
        return (Element) matches.item(0);
    }

    private static double calibrationNumber(Element element, String name) {
        double value = Double.parseDouble(element.getAttribute(name));
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("Nonfinite calibration attribute: " + name);
        }
        return value;
    }

    private static int calibrationInteger(Element element, String name) {
        return Integer.parseInt(element.getAttribute(name));
    }

    private static boolean calibrationBoolean(Element root, String name) {
        String value = calibrationElement(root, name).getTextContent().trim();
        if (!(value.equals("true") || value.equals("false"))) {
            throw new IllegalArgumentException("Invalid calibration boolean: " + name);
        }
        return Boolean.parseBoolean(value);
    }

    private Document calibrationDocument(String xml) throws IOException {
        calibrationInitReceipt = null;
        // Do not reinterpret ordinary MineRL missions.
        if (!xml.contains("Calib_")) return null;
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setNamespaceAware(true);
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
            factory.setXIncludeAware(false);
            factory.setExpandEntityReferences(false);
            Document doc = factory.newDocumentBuilder().parse(new InputSource(new StringReader(xml)));
            Element root = doc.getDocumentElement();
            if (!calibrationElement(root, "Summary").getTextContent().trim().startsWith("Calib_")) {
                return null;
            }
            if (root.getElementsByTagNameNS("*", "AgentSection").getLength() != 1) {
                throw new IllegalArgumentException("Calibration supports exactly one player");
            }
            return doc;
        } catch (Exception error) {
            throw new IOException("Invalid calibration mission XML", error);
        }
    }

    private CompletableFuture<JsonObject> scheduleCalibrationInit(Minecraft mc, Document doc) {
        CompletableFuture<JsonObject> ready = new CompletableFuture<>();
        MinecraftServer server = mc.getIntegratedServer();
        if (server == null || !mc.gameSettings.syncIntegratedServer) {
            ready.completeExceptionally(new IllegalStateException(
                    "Calibration requires MineRL's local synchronous integrated server"));
            return ready;
        }
        // In this MineRL build the synchronous server is owned by the client thread.
        // Queue there, then let the EXISTING bootstrap no-ops advance the loop.
        // Waiting here before a bootstrap action would deadlock ReplaySender.tick().
        mc.execute(() -> {
            try {
                if (!server.isOnExecutionThread()) {
                    throw new IllegalStateException("Not on the integrated-server execution thread");
                }
                JsonObject receipt = initializeCalibrationOnServer(server, doc);
                calibrationInitReceipt = receipt;
                System.out.println("CALIB_INIT " + receipt.toString());
                ready.complete(receipt);
            } catch (Throwable error) {
                ready.completeExceptionally(error);
            }
        });
        return ready;
    }

    private JsonObject initializeCalibrationOnServer(MinecraftServer server, Document doc) {
        Element root = doc.getDocumentElement();
        if (server.getPlayerList().getPlayers().size() != 1) {
            throw new IllegalStateException("Calibration expected one integrated-server player");
        }
        ServerPlayerEntity player = server.getPlayerList().getPlayers().get(0);
        ServerWorld world = player.getServerWorld();
        Element drawing = calibrationElement(root, "DrawingDecorator");
        Element placement = calibrationElement(root, "Placement");
        Element food = calibrationElement(root, "StartingFood");
        Element health = calibrationElement(root, "StartingHealth");
        int initialFood = calibrationInteger(food, "food");
        float saturation = (float) calibrationNumber(food, "foodSaturation");
        float initialHealth = (float) calibrationNumber(health, "health");
        float maxHealth = (float) calibrationNumber(health, "maxHealth");
        double px = calibrationNumber(placement, "x");
        double py = calibrationNumber(placement, "y");
        double pz = calibrationNumber(placement, "z");
        float yaw = (float) calibrationNumber(placement, "yaw");
        float pitch = (float) calibrationNumber(placement, "pitch");
        if (initialFood < 0 || initialFood > 20 || saturation < 0 || saturation > initialFood
                || maxHealth != 20.0f || initialHealth <= 0 || initialHealth > maxHealth
                || Math.abs(px) > 32 || Math.abs(pz) > 32 || py < 1 || py > 100
                || Math.abs(pitch) > 90) {
            throw new IllegalArgumentException("Calibration player state outside supported bounds");
        }
        long time = Long.parseLong(calibrationElement(root, "StartTime").getTextContent().trim());
        boolean passage = calibrationBoolean(root, "AllowPassageOfTime");
        boolean spawning = calibrationBoolean(root, "AllowSpawning");
        if (!calibrationElement(root, "Weather").getTextContent().trim().equals("clear")) {
            throw new IllegalArgumentException("Calibration supports clear weather only");
        }

        // Build the final requested state first. Overlapping cuboids obey XML order.
        // Only a small explicit native-block whitelist is supported, never silent fallback to air.
        Set<String> permitted = new HashSet<>(Arrays.asList(
                "air", "stone", "grass_block", "oak_leaves", "oak_log", "dirt"));
        Map<BlockPos, BlockState> expected = new LinkedHashMap<>();
        JsonArray primitives = new JsonArray();
        long requestedWrites = 0;
        NodeList nodes = drawing.getChildNodes();
        for (int i = 0; i < nodes.getLength(); i++) {
            Node child = nodes.item(i);
            if (!(child instanceof Element)) continue;
            Element element = (Element) child;
            String tag = element.getLocalName();
            if (!(tag.equals("DrawBlock") || tag.equals("DrawCuboid"))) {
                throw new IllegalArgumentException("Unsupported calibration primitive: " + tag);
            }
            String type = element.getAttribute("type");
            if (type.startsWith("minecraft:")) type = type.substring("minecraft:".length());
            if (!permitted.contains(type)) {
                throw new IllegalArgumentException("Unsupported calibration block: " + type);
            }
            ResourceLocation blockId = new ResourceLocation("minecraft", type);
            if (!Registry.BLOCK.containsKey(blockId)) {
                throw new IllegalArgumentException("Unknown native block: " + type);
            }
            Block block = Registry.BLOCK.getOrDefault(blockId);
            int x1, y1, z1, x2, y2, z2;
            if (tag.equals("DrawBlock")) {
                x1 = x2 = calibrationInteger(element, "x");
                y1 = y2 = calibrationInteger(element, "y");
                z1 = z2 = calibrationInteger(element, "z");
            } else {
                x1 = calibrationInteger(element, "x1"); x2 = calibrationInteger(element, "x2");
                y1 = calibrationInteger(element, "y1"); y2 = calibrationInteger(element, "y2");
                z1 = calibrationInteger(element, "z1"); z2 = calibrationInteger(element, "z2");
            }
            if (Math.abs(x1) > 32 || Math.abs(x2) > 32 || Math.abs(z1) > 32 || Math.abs(z2) > 32
                    || y1 < 1 || y2 < 1 || y1 > 100 || y2 > 100 || x1 > x2 || y1 > y2 || z1 > z2) {
                throw new IllegalArgumentException("Calibration drawing outside supported bounds");
            }
            requestedWrites += (long) (x2-x1+1) * (y2-y1+1) * (z2-z1+1);
            if (requestedWrites > 100000) throw new IllegalArgumentException("Calibration drawing too large");
            for (int x = x1; x <= x2; x++) for (int y = y1; y <= y2; y++) for (int z = z1; z <= z2; z++) {
                expected.put(new BlockPos(x, y, z), block.getDefaultState());
            }
            JsonObject primitive = new JsonObject();
            primitive.addProperty("primitive", tag); primitive.addProperty("type", "minecraft:" + type);
            primitive.addProperty("x1", x1); primitive.addProperty("y1", y1); primitive.addProperty("z1", z1);
            primitive.addProperty("x2", x2); primitive.addProperty("y2", y2); primitive.addProperty("z2", z2);
            primitives.add(primitive);
        }
        if (expected.isEmpty()) throw new IllegalArgumentException("Empty calibration drawing");

        // Rules and geometry are initialization only. Mining, hunger, use and pickup stay native.
        world.getGameRules().get(GameRules.DO_MOB_SPAWNING).set(spawning, server);
        world.getGameRules().get(GameRules.DO_DAYLIGHT_CYCLE).set(passage, server);
        world.getGameRules().get(GameRules.DO_WEATHER_CYCLE).set(false, server);
        world.setDayTime(time);
        // Vanilla 1.16.5 SRG method; javap signature: (int clearTicks, int rainTicks,
        // boolean raining, boolean thundering). No name exists in this snapshot mapping.
        world.func_241113_a_(1000000, 0, false, false);
        for (Map.Entry<BlockPos, BlockState> entry : expected.entrySet()) {
            // Native setBlockState, with client update and suppressed initialization drops.
            // These flags apply to these writes only, not later policy mining or physics.
            world.setBlockState(entry.getKey(), entry.getValue(), 2 | 16 | 32);
        }
        int verified = 0;
        for (Map.Entry<BlockPos, BlockState> entry : expected.entrySet()) {
            if (world.getBlockState(entry.getKey()).getBlock() != entry.getValue().getBlock()) {
                throw new IllegalStateException("Calibration block write failed at " + entry.getKey());
            }
            verified++;
        }

        player.inventory.clear();
        AgentStart.Inventory inventory = getAgentStart(missionInit).getInventory();
        if (inventory != null) inventory.getInventoryObject().forEach(entry -> {
            ResourceLocation itemId = new ResourceLocation(entry.getValue().getType());
            int slot = entry.getValue().getSlot();
            int count = entry.getValue().getQuantity();
            if (!Registry.ITEM.containsKey(itemId) || slot < 0 || slot > 35 || count < 1 || count > 64) {
                throw new IllegalArgumentException("Invalid calibration starting inventory");
            }
            player.inventory.setInventorySlotContents(slot, new ItemStack(Registry.ITEM.getOrDefault(itemId), count));
        });
        player.inventory.currentItem = 0;
        player.getAttribute(Attributes.MAX_HEALTH).setBaseValue(maxHealth);
        player.setHealth(initialHealth);
        player.getFoodStats().setFoodLevel(initialFood);
        player.getFoodStats().setFoodSaturationLevel(saturation);
        player.setMotion(0.0, 0.0, 0.0);
        player.fallDistance = 0.0f;
        // Authoritative server teleport, with the normal position packet to the client.
        player.connection.setPlayerLocation(px, py, pz, yaw, pitch);
        player.sendContainerToPlayer(player.container);
        player.connection.sendPacket(new SHeldItemChangePacket(0));
        player.connection.sendPacket(new SUpdateHealthPacket(player.getHealth(),
                player.getFoodStats().getFoodLevel(), player.getFoodStats().getSaturationLevel()));

        JsonObject receipt = new JsonObject();
        receipt.addProperty("schema", "calibration_native_init_v1");
        receipt.addProperty("summary", calibrationElement(root, "Summary").getTextContent().trim());
        receipt.addProperty("applied_once", true);
        receipt.addProperty("applied", true);
        receipt.addProperty("generator", "native world generator with local drawn arena; FlatWorldGenerator not implemented");
        receipt.addProperty("server_execution_thread", Thread.currentThread().getName());
        receipt.addProperty("server_thread_verified", server.isOnExecutionThread());
        receipt.addProperty("server_tick", server.getTickCounter());
        receipt.addProperty("draw_primitives", primitives.size());
        receipt.addProperty("requested_writes_including_overlap", requestedWrites);
        receipt.addProperty("final_unique_positions", expected.size());
        receipt.addProperty("verified_block_types", verified);
        receipt.addProperty("block_mismatches", 0);
        receipt.addProperty("verification_failures", 0);
        receipt.add("drawing", primitives);
        receipt.addProperty("x", player.getPosX()); receipt.addProperty("y", player.getPosY()); receipt.addProperty("z", player.getPosZ());
        receipt.addProperty("yaw", player.rotationYaw); receipt.addProperty("pitch", player.rotationPitch);
        receipt.addProperty("health", player.getHealth()); receipt.addProperty("max_health", player.getMaxHealth());
        receipt.addProperty("food", player.getFoodStats().getFoodLevel());
        receipt.addProperty("saturation", player.getFoodStats().getSaturationLevel());
        receipt.addProperty("selected_hotbar_slot", player.inventory.currentItem);
        receipt.addProperty("selected_slot", player.inventory.currentItem);
        receipt.addProperty("day_time", world.getDayTime());
        receipt.addProperty("do_daylight_cycle", world.getGameRules().getBoolean(GameRules.DO_DAYLIGHT_CYCLE));
        receipt.addProperty("do_mob_spawning", world.getGameRules().getBoolean(GameRules.DO_MOB_SPAWNING));
        receipt.addProperty("do_weather_cycle", world.getGameRules().getBoolean(GameRules.DO_WEATHER_CYCLE));
        receipt.addProperty("weather", "clear");
        receipt.addProperty("phase", "initialization before existing bootstrap frames; evaluator must check reset observation too");
        return receipt;
    }

    private void requireCalibrationReady(CompletableFuture<JsonObject> ready) throws IOException, InterruptedException {
        if (ready == null) return;
        try {
            // Existing bootstrap actions must already have advanced the owning thread.
            // Never silently begin a trial if initialization failed or has not finished.
            ready.get(5, TimeUnit.SECONDS);
        } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException error) {
            throw new IOException("Native calibration initialization failed", error);
        }
    }
'''

INVENTORY_OLD = """        mc.execute(() -> setAgentInventory(mc.player, missionInit));
        mc.execute(() -> setAgentPosition(mc.player, missionInit));"""
INVENTORY_NEW = """        CompletableFuture<JsonObject> calibrationReady = null;
        if (calibrationXml != null) {
            calibrationReady = scheduleCalibrationInit(mc, calibrationXml);
        } else {
            mc.execute(() -> setAgentInventory(mc.player, missionInit));
            mc.execute(() -> setAgentPosition(mc.player, missionInit));
        }"""


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one patch anchor, got {count}: {old[:110]!r}")
    return text.replace(old, new, 1)


def patched_source(original: str) -> str:
    if MARKER in original:
        raise ValueError("Native calibration initialization is already installed")
    text = replace_once(original, "public class EnvServer {", IMPORTS + "\npublic class EnvServer {")
    text = replace_once(text, "        missionInit = MissionSpec.decodeMissionInit(command);",
                        "        missionInit = MissionSpec.decodeMissionInit(command);\n"
                        "        final Document calibrationXml = calibrationDocument(command);")
    text = replace_once(text, INVENTORY_OLD, INVENTORY_NEW)
    text = replace_once(text, "        mc.execute( () -> {\n                    Pos startV = getAgentStart(missionInit).getVelocity();",
                        "        requireCalibrationReady(calibrationReady);\n\n"
                        "        mc.execute( () -> {\n                    Pos startV = getAgentStart(missionInit).getVelocity();")
    text = replace_once(text, '        infoJson.addProperty("isGuiOpen", mc.currentScreen != null);',
                        '        if (calibrationInitReceipt != null) infoJson.add("calibration_init", calibrationInitReceipt);\n'
                        '        infoJson.addProperty("isGuiOpen", mc.currentScreen != null);')
    text = replace_once(text, "    private void setAgentInventory(ClientPlayerEntity player, MissionInit missionInit) {",
                        METHODS + "\n    private void setAgentInventory(ClientPlayerEntity player, MissionInit missionInit) {")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Generated EnvServer.java after the official MineRL patch")
    parser.add_argument("--apply", action="store_true", help="Write the reviewed change; without this flag only print the diff")
    parser.add_argument("--report-dir", type=Path, help="Save source hashes and patch here when applying")
    args = parser.parse_args()
    original = args.source.read_text()
    updated = patched_source(original)
    diff = "".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                      fromfile=str(args.source), tofile=str(args.source)))
    if not args.apply:
        print(diff, end="")
        return
    report_dir = args.report_dir or args.source.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "EnvServer.before-calibration.java").write_text(original)
    (report_dir / "EnvServer.calibration.patch").write_text(diff)
    record = {
        "source": str(args.source.resolve()),
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "scope": "Calib_ mission initialization and evaluator receipt only",
        "policy_modified": False,
        "resource_mechanics_modified": False,
        "flat_world_generator_implemented": False,
        "java_compilation_verified_by_this_script": False,
        "minecraft": "1.16.5",
        "mcp_snapshot": "20201028-1.16.3",
        "mapping_source": "https://maven.minecraftforge.net/de/oceanlabs/mcp/mcp_snapshot/20201028-1.16.3/mcp_snapshot-20201028-1.16.3.zip",
    }
    (report_dir / "EnvServer.calibration.provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    args.source.write_text(updated)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
