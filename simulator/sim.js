// Same dynamics as inverter-rl/env.py. One random number per hour, whatever the action,
// so every policy run with the same seed faces the same power cuts.
(function (root) {
  const MAX_BATTERY = 10, MAX_OFF = 3, CHARGE = 0, HOLD = 1, DISCHARGE = 2;

  function period(hour) {
    if (hour >= 18 && hour <= 21) return "peak";
    if (hour >= 10 && hour <= 17) return "solar";
    return "normal";
  }

  function batteryStep(battery, off, action) {
    if (off) return battery === 0 ? [0, 0, true] : [battery - 1, 0, false];
    if (action === CHARGE && battery < MAX_BATTERY) return [battery + 1, 2, false];
    if (action === DISCHARGE && battery > 0) return [battery - 1, 0, false];
    return [battery, 1, false]; // hold (also charge at full, discharge at empty)
  }

  function changeProb(data, hour, off, pattern) {
    return off === 0 ? data.cut[pattern][period(hour)] : data.restore[pattern][off - 1];
  }

  function nextOff(off, changed) {
    if (off === 0) return changed ? 1 : 0;
    return changed ? 0 : Math.min(off + 1, MAX_OFF);
  }

  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function action(policy, battery, hour, off) {
    return policy.charCodeAt(battery * 96 + hour * 4 + off) - 48;
  }

  // Advances the shared grid by one hour and every lane by one step.
  // world: {hour, off, random}; lanes: [{policy, battery, bill, blackouts, score}]
  function step(data, world, lanes, pattern, penalty) {
    const price = data.price[period(world.hour)];
    const record = lanes.map((lane) => {
      const a = action(lane.policy, lane.battery, world.hour, world.off);
      const [battery, units, blackout] = batteryStep(lane.battery, world.off, a);
      const bill = price * units;
      lane.battery = battery;
      lane.bill += bill;
      lane.blackouts += blackout ? 1 : 0;
      lane.score += -bill - penalty * (blackout ? 1 : 0);
      return { action: a, battery, blackout, bill };
    });
    const hour = world.hour, off = world.off;
    const changed = world.random() < changeProb(data, world.hour, world.off, pattern);
    world.off = nextOff(world.off, changed);
    world.hour = (world.hour + 1) % 24;
    return { hour, off, price, lanes: record };
  }

  root.Sim = { period, step, rng, action, CHARGE, HOLD, DISCHARGE, MAX_BATTERY };
})(typeof window !== "undefined" ? window : globalThis);
