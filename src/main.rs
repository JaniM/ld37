
#![feature(slice_patterns)]

extern crate rand;
extern crate sdl2;
extern crate sdl2_ttf;
extern crate clock_ticks;
extern crate itertools;

use std::ops;

use sdl2::render::{Renderer, TextureQuery, BlendMode};
use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::{MouseState};
use sdl2::rect::{Point, Rect};
use sdl2_ttf::{Font};

use rand::distributions::{IndependentSample};
use rand::{Rng};

const STOP_LIMIT: f64 = 10.0;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Pointf64 {
    pub x: f64,
    pub y: f64
}

impl Pointf64 {
    pub fn new(x: f64, y: f64) -> Self {
        Pointf64 {
            x: x,
            y: y
        }
    }
    pub fn from_i32(x: i32, y: i32) -> Self {
        Self::new(x as f64, y as f64)
    }

    pub fn from_sdl(p: Point) -> Self {
        Self::new(p.x() as f64, p.y() as f64)
    }

    pub fn of_rad(rad: f64) -> Self {
        Self::new(rad.cos(), rad.sin())
    }

    pub fn rotate(&self, rad: f64) -> Self {
    	let cos = rad.cos();
    	let sin = rad.sin();
    	Self::new(self.x * cos - self.y * sin, self.x * sin + self.y * cos)
    }

    pub fn normal_rhs(&self) -> Self {
    	Self::new(-self.y, self.x)
    }

    pub fn normal_lhs(&self) -> Self {
    	Self::new(self.y, -self.x)
    }

    pub fn reflect(self, normal: Pointf64) -> Self {
    	self - normal.unit()*(2.0*(self * normal))
    }

    pub fn as_sdl(&self) -> Point {
        Point::new(self.x as i32, self.y as i32)
    }

    pub fn length(&self) -> f64 {
        (self.x*self.x + self.y*self.y).sqrt()
    }

    pub fn angle(&self) -> f64 {
        f64::atan2(self.y, self.x)
    }

    pub fn angle_to(&self, other: Pointf64) -> f64 {
        f64::atan2(other.y - self.y, other.x - self.x)
    }

    pub fn unit(&self) -> Pointf64 {
        self / self.length()
    }
}

impl ops::Add for Pointf64 {
    type Output = Pointf64;
    fn add(self, rhs: Pointf64) -> Pointf64 {
        Pointf64::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl ops::Sub for Pointf64 {
    type Output = Pointf64;
    fn sub(self, rhs: Pointf64) -> Pointf64 {
        Pointf64::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl ops::Mul for Pointf64 {
    type Output = f64;
    fn mul(self, rhs: Pointf64) -> f64 {
        (self.x * rhs.x) + (self.y * rhs.y)
    }
}

impl ops::Mul<f64> for Pointf64 {
    type Output = Pointf64;
    fn mul(self, rhs: f64) -> Pointf64 {
        Pointf64::new(self.x * rhs, self.y * rhs)
    }
}

impl ops::Div<f64> for Pointf64 {
    type Output = Pointf64;
    fn div(self, rhs: f64) -> Pointf64 {
        Pointf64::new(self.x / rhs, self.y / rhs)
    }
}

impl<'a> ops::Add for &'a Pointf64 {
    type Output = Pointf64;
    fn add(self, rhs: &Pointf64) -> Pointf64 {
        Pointf64::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<'a> ops::Sub for &'a Pointf64 {
    type Output = Pointf64;
    fn sub(self, rhs: &Pointf64) -> Pointf64 {
        Pointf64::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<'a> ops::Mul for &'a Pointf64 {
    type Output = f64;
    fn mul(self, rhs: &Pointf64) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }
}

impl<'a> ops::Mul<f64> for &'a Pointf64 {
    type Output = Pointf64;
    fn mul(self, rhs: f64) -> Pointf64 {
        Pointf64::new(self.x * rhs, self.y * rhs)
    }
}

impl<'a> ops::Div<f64> for &'a Pointf64 {
    type Output = Pointf64;
    fn div(self, rhs: f64) -> Pointf64 {
        Pointf64::new(self.x / rhs, self.y / rhs)
    }
}


pub fn sqrt(x: i32) -> i32 {
    (x as f32).sqrt() as i32
}

pub fn dist_to_segment(v: Pointf64, w: Pointf64, p: Pointf64) -> f64 {
	segment_projection(v, w, p).length()
}

pub fn segment_projection(v: Pointf64, w: Pointf64, p: Pointf64) -> Pointf64 {
    let l2 = (v.x - w.x).powi(2) + (v.y - w.y).powi(2); // dist2(v, w);
    if l2 == 0.0 {
  	    return p - w; // dist2(p, v);
    }
    let t = f64::max(0.0, f64::min(1.0, ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2));
    return p - Pointf64::new(v.x + t * (w.x - v.x), v.y + t * (w.y - v.y));
}


trait SceneObject {
    fn update(&mut self, game: &mut GameState, delta: f64);
    fn render(&self, game: &mut GameState, renderer: &mut Renderer, delta: f64);
    fn handle_event(&mut self, _game: &mut GameState, _event: Event) {

    }
    fn quit_on_esc(&self) -> bool { true }
}

trait Collide<Other> {
    fn collides(&self, other: &Other) -> usize;
    fn reflect(&self, circle: &Other, delta: f64) -> (Pointf64, Pointf64);
}

trait Effect {
    fn update(&mut self, delta: f64) -> EffectChange;
    fn render(&self, renderer: &mut Renderer, delta: f64);
}

#[derive(Debug, Copy, Clone)]
enum EffectChange {
    None,
    Ended
}

#[derive(Debug, Copy, Clone)]
struct BackgroundEffect {
    start: (u8, u8, u8),
    end: (u8, u8, u8),
    width: i32,
    direction: bool,
    duration: f64,
    time: f64,
    alpha: u8
}

impl BackgroundEffect {
    fn new(start: (u8, u8, u8), end: (u8, u8, u8), width: i32, direction: bool, duration: f64, alpha: u8) -> Self {
        BackgroundEffect {
            start: start,
            end: end,
            width: width,
            direction: direction,
            duration: duration,
            time: 0.0,
            alpha: alpha
        }
    }
}

impl Effect for BackgroundEffect {
    fn update(&mut self, delta: f64) -> EffectChange {
        self.time += delta;
        return if self.time >= self.duration {
            EffectChange::Ended
        } else {
            EffectChange::None
        }
    }
    fn render(&self, renderer: &mut Renderer, _delta: f64) {
        let pos = if self.direction {
            ((800.0 + self.width as f64) * (self.time / self.duration)) as i32
        } else {
            800 - ((800.0 + self.width as f64) * (self.time / self.duration)) as i32
        };
        for i in 0..self.width {
            let shade = 1.0 - i as f64 / self.width as f64;
            let r = self.start.0 + ((self.end.0 - self.start.0) as f64 * shade) as u8;
            let g = self.start.1 + ((self.end.1 - self.start.1) as f64 * shade) as u8;
            let b = self.start.2 + ((self.end.2 - self.start.2) as f64 * shade) as u8;
            renderer.set_draw_color(Color::RGBA(r, g, b, self.alpha));
            if self.direction {
                renderer.draw_line(Point::new(pos - i, 0), Point::new(pos - i, 600)).unwrap();
            } else {
                renderer.draw_line(Point::new(pos + i, 0), Point::new(pos + i, 600)).unwrap();
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SpawnEffect {
    lifetimes: Vec<f64>,
    circles: Vec<Circle>,
    walls: Vec<Wall>,
    color: Color
}

impl SpawnEffect {
    fn new(x: f64, y: f64, color: Color, mincount: i32, maxcount: i32, minlife: f64, maxlife: f64, minspeed: f64, maxspeed: f64, walls: Vec<Wall>) -> Self {
        let mut rng = rand::thread_rng();
        let count = rng.gen_range(mincount, maxcount);
        SpawnEffect {
            lifetimes: (0..count).map(|_| rng.gen_range(minlife, maxlife)).collect(),
            circles: (0..count).map(|_| {
                Circle::new(x, y, rng.gen_range(1, 3) as f64, Pointf64::of_rad(rng.gen_range(-std::f64::consts::PI, std::f64::consts::PI)) * rng.gen_range(minspeed, maxspeed), 0.8)
            }).collect(),
            walls: walls,
            color: color
        }
    }
}

impl Effect for SpawnEffect {
    fn update(&mut self, delta: f64) -> EffectChange {
        let mut i = 0;
        while i < self.circles.len() {
            self.lifetimes[i] -= delta;
            if self.lifetimes[i] <= 0.0 {
                self.circles.remove(i);
                self.lifetimes.remove(i);
                continue;
            } else {
                self.circles[i].apply_velocity(delta, &self.walls, 800.0, 600.0);
                i += 1;
            }
        }
        return if self.circles.len() == 0 {
            EffectChange::Ended
        } else {
            EffectChange::None
        }
    }
    fn render(&self, renderer: &mut Renderer, _delta: f64) {
        renderer.set_draw_color(self.color);
        for circle in &self.circles {
            draw_circle(renderer, circle.point.as_sdl(), circle.radius as i32);
        }
    }
}

enum GameStateChange {
    SwitchScene(Box<SceneObject>)
}

struct Inputs {
    mouse_x: i32,
    mouse_y: i32,
    mouse: MouseState,
    last_mouse: MouseState
}

struct GameState {
    changes: Vec<GameStateChange>,
    inputs: Inputs,
    fps: f64,
    info_font: Font,
    score_font: Font,
    text_font: Font
}

impl Inputs {
    fn new() -> Self {
         Inputs {
            mouse_x: 0,
            mouse_y: 0,
            mouse: MouseState::from_flags(0),
            last_mouse: MouseState::from_flags(0)
        }
    }
}

impl GameState {
    fn switch_scene<'a, T: SceneObject>(&'a mut self, new_scene: T) where T: 'static {
        self.changes.push(GameStateChange::SwitchScene(Box::new(new_scene) as Box<SceneObject>));
    }
}

struct PlayState {
    player: Player,
    aim: Targeting,
    targets: Vec<Circle>,
    walls: Vec<Wall>,
    spawn_timer: f64,
    spawn_time_limit: f64,
    trail_time: f64,
    time: f64,
    effects: Vec<Box<Effect>>,
    middle_effects: Vec<Box<Effect>>,
    bgtimer: f64
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct Targeting {
    angle: f64,
    length: f64
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct Player {
    circle: Circle,
    trail: Trail,
    circling: Option<(bool, Pointf64)>
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct Trail {
    circles: Vec<(f64, Circle)>
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct Circle {
    point: Pointf64,
    radius: f64,
    drag: f64,
    velocity: Pointf64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Wall {
    points: Vec<Pointf64>
}

impl Wall {
    fn new(coords: &[f64]) -> Self {
        let mut w = Wall {
            points: coords.chunks(2).map(|pair| Pointf64::new(pair[0], pair[1])).collect()
        };
        if w.points.len() > 2 {
            let first = w.points[0];
            w.points.push(first);
        }
        w
    }

    fn draw(&self, renderer: &mut Renderer) {
        for pair in self.points.windows(2) {
            renderer.draw_line(pair[0].as_sdl(), pair[1].as_sdl() ).unwrap();
        }
    }

    fn select_closest(&self, point: Pointf64) -> (Pointf64, Pointf64) {
        let pair = self.points.windows(2).min_by_key(|pair| dist_to_segment(pair[0], pair[1], point) as i32).unwrap();
        (pair[0], pair[1])
    }

    fn select_closest_facing(&self, point: Pointf64) -> Option<(Pointf64, Pointf64)> {
        self.points
            .windows(2)
            .filter(|pair| {
                let projection = segment_projection(pair[0], pair[1], point);
                (projection * (pair[1] - pair[0])) as i32 == 0 && (pair[1] - pair[0]).normal_lhs() * projection > 0.0
            })
            .min_by_key(|pair| dist_to_segment(pair[0], pair[1], point) as i32)
            .map(|pair| (pair[0], pair[1]))
    }
}

impl Collide<Circle> for Wall {
    fn collides(&self, circle: &Circle) -> usize {
        self.points.windows(2).filter(|pair| dist_to_segment(pair[0], pair[1], circle.point) < circle.radius).count()
    }

    fn reflect(&self, circle: &Circle, delta: f64) -> (Pointf64, Pointf64) {
        let (a, b) = self.select_closest(circle.point);
        let projection = segment_projection(a, b, circle.point);
        if (projection * (b - a)) as i32 != 0 {
            let normal = (b - a).normal_lhs().unit();
            if projection * normal < 0.0 || projection.length() < circle.radius {
                return (circle.velocity * delta, circle.velocity);
            }
            let corner = if (circle.point - b).length() < (circle.point - a).length() { b } else { a };
            let step_vel = circle.velocity.unit() * (projection.length() - circle.radius);
            let normal = (circle.point - corner).unit();
            (step_vel, circle.velocity.reflect(normal))
        } else {
            let (a, b) = self.select_closest_facing(circle.point).unwrap_or((a, b));
            let projection = segment_projection(a, b, circle.point);
            let normal = (b - a).normal_lhs().unit();
            let step_vel = circle.velocity.unit()*(dist_to_segment(a, b, circle.point) - circle.radius);
            if projection * normal < 0.0 || projection.length() < circle.radius {
                return (circle.velocity * delta, circle.velocity);
            }
            (step_vel, circle.velocity.reflect(normal))
        }
    }
}

impl Targeting {
    fn new(angle: f64, length: f64) -> Self {
        Targeting {
            angle: angle,
            length: length
        }
    }
}

impl ops::Sub<f64> for Targeting {
    type Output = Targeting;
    fn sub(self, rhs: f64) -> Targeting {
        Targeting::new(self.angle, self.length - rhs)
    }
}

impl ops::Mul<f64> for Targeting {
    type Output = Targeting;
    fn mul(self, rhs: f64) -> Targeting {
        Targeting::new(self.angle, self.length * rhs)
    }
}

impl Circle {
    fn new(x: f64, y: f64, radius: f64, velocity: Pointf64, drag: f64) -> Self {
        Circle {
            point: Pointf64::new(x, y),
            velocity: velocity,
            drag: drag,
            radius: radius
        }
    }

    fn random(width: f64, height: f64, minspeed: f64, maxspeed: f64, minrad: f64, maxrad: f64, drag: f64) -> Self {
        let mut rng = rand::thread_rng();
        let widthd = rand::distributions::Range::new(0.0f64, width);
        let heightd = rand::distributions::Range::new(0.0f64, height);
        let veld = rand::distributions::Range::new(minspeed, maxspeed);
        let angled = rand::distributions::Range::new(0.0f64, 2.0 * std::f64::consts::PI);
        let radiusd = rand::distributions::Range::new(minrad, maxrad);
        Self::new(widthd.ind_sample(&mut rng), heightd.ind_sample(&mut rng), radiusd.ind_sample(&mut rng), Pointf64::of_rad(angled.ind_sample(&mut rng)) * veld.ind_sample(&mut rng), drag)
    }

    fn random_n(n: i32, width: f64, height: f64, minspeed: f64, maxspeed: f64, minrad: f64, maxrad: f64, drag: f64) -> Vec<Self> {
        (0..n).map(|_| Self::random(width, height, minspeed, maxspeed, minrad, maxrad, drag)).collect()
    }

    fn apply_velocity(&mut self, delta: f64, walls: &Vec<Wall>, width: f64, height: f64) {
        let vel = self.velocity * delta;
        let mut step_vel = vel;
        let new_pos = self.point + vel;
        if vel.x > 0.0 && new_pos.x + self.radius >= width {
            step_vel.x = width - self.point.x - self.radius;
            self.velocity.x *= -1.0;
        } else if vel.x < 0.0 && new_pos.x - self.radius < 0.0 {
            step_vel.x = self.point.x - self.radius;
            self.velocity = self.velocity.reflect(Pointf64::new(1.0, 0.0));
        }
        if vel.y > 0.0 && new_pos.y + self.radius >= height {
            step_vel.y = height - self.point.y- self.radius;
            self.velocity.y *= -1.0;
        } else if vel.y < 0.0 && new_pos.y - self.radius < 0.0 {
            step_vel.y = self.point.y - self.radius;
            self.velocity.y *= -1.0;
        }
        loop {
            let mut hit = false;
            for wall in walls.iter() {
                self.point = self.point + step_vel;
                let hits = wall.collides(self);
                if hits > 0 {
                    self.point = self.point - step_vel;
                    let (step, velocity) = wall.reflect(self, delta);
                    if velocity == self.velocity { break; }
                    step_vel = velocity * delta;
                    self.point = self.point + step;
                    self.velocity = velocity;
                    hit = true;
                    break;
                } else {
                    self.point = self.point - step_vel;
                }
            }
            if !hit { break; }
        }
        self.point = self.point + step_vel;
        self.velocity = self.velocity - self.velocity * self.drag * delta;
    }

    fn is_colliding(&self, other: &Circle) -> bool {
        (self.point - other.point).length().abs() < self.radius + other.radius
    }
}

impl PlayState {
    fn initial(target_count: i32, spawn_time_limit: f64, trail_time: f64) -> PlayState {
        println!("Started game with {} targets, {} spawn time and {} trail", target_count, spawn_time_limit, trail_time);
        PlayState {
            player: Player {
                circle: Circle::new(400.0, 300.0, 10.0, Pointf64::new(0.0, 0.0), 0.9),
                trail: Trail { circles: vec![] },
                circling: None
            },
            aim: Targeting::new(0.0, 0.0),
            targets: Circle::random_n(target_count, 800.0, 600.0, 50.0, 250.0, 5.0, 10.0, 0.0),
            walls: vec![Wall::new(&[200.0, 200.0, 250.0, 200.0, 250.0, 400.0, 200.0, 400.0]),
                        Wall::new(&[550.0, 200.0, 600.0, 200.0, 600.0, 400.0, 550.0, 400.0])],
            spawn_timer: 0.0,
            spawn_time_limit: spawn_time_limit,
            trail_time: trail_time,
            time: 0.0,
            effects: vec![],
            middle_effects: vec![],
            bgtimer: 1.0
        }
    }
}

impl SceneObject for PlayState {
    fn update(&mut self, game: &mut GameState, delta: f64) {
        let max_strength = 300.0;
        let strength_scale = 1.0;

        // Mouse coordinate as a handy vector :)
        let mouse_p = Pointf64::from_i32(game.inputs.mouse_x, game.inputs.mouse_y);
        //let clicked = !game.inputs.last_mouse.left() && game.inputs.mouse.left();

        self.time += delta;
        self.spawn_timer += delta;
        self.bgtimer -= delta;

        // Align and limit targeting
        let diff = mouse_p - self.player.circle.point;
        let angle = diff.angle();
        let length = f64::min(max_strength, diff.length());
        self.aim = Targeting::new(angle, length);

        // Apply velocity on click
        if game.inputs.mouse.left() { // && self.player.circling.is_some() {
            let target = Targeting::new(angle, length) * strength_scale;
            let dist = target.length;
            let v = Pointf64::of_rad(target.angle) * dist / 10.0;
            self.player.circle.velocity = self.player.circle.velocity + v;
        }
        if !game.inputs.mouse.left() {
            self.player.circling = None;
        }

        self.player.circle.apply_velocity(delta, &self.walls, 800.0, 600.0);

        if self.spawn_timer > self.spawn_time_limit {
            self.spawn_timer -= self.spawn_time_limit;
            let target = Circle::random(800.0, 600.0, 50.0, 250.0, 5.0, 10.0, 0.0);
            if self.spawn_time_limit <= 2.0 {
                self.middle_effects.push(Box::new(SpawnEffect::new(target.point.x, target.point.y, Color::RGB(255, 255, 255), 5, 10, 1.0, 3.0, 50.0, 300.0, self.walls.clone())) as Box<Effect>);
            } else {
                self.middle_effects.push(Box::new(SpawnEffect::new(target.point.x, target.point.y, Color::RGB(255, 255, 255), 10, 20, 1.0, 3.0, 50.0, 300.0, self.walls.clone())) as Box<Effect>);
            }
            self.targets.push(target);
        }

        // Update all targets.
        let mut killed = vec![];
        'targetloop: for (i, circle) in self.targets.iter_mut().enumerate() {
            circle.apply_velocity(delta, &self.walls, 800.0, 600.0);
            if circle.is_colliding(&self.player.circle) {
                killed.push(i);
                continue 'targetloop;
            }
            for &(_a, ref t) in &self.player.trail.circles {
                if circle.is_colliding(t) {
                    killed.push(i);
                    continue 'targetloop;
                }
            }
        }
        let mut ckilled = 0;
        for i in killed {
            {
                let circle = self.targets[i - ckilled];
                self.middle_effects.push(Box::new(SpawnEffect::new(circle.point.x, circle.point.y, Color::RGB(255, 0, 0), 5, 10, 1.0, 3.0, 50.0, 300.0, self.walls.clone())) as Box<Effect>);
            }
            self.targets.remove(i - ckilled);
            ckilled += 1;
        }

        self.player.trail.circles.push((self.trail_time, self.player.circle));
        let mut i = 0;
        while i < self.player.trail.circles.len() {
            self.player.trail.circles[i].0 -= delta;
            if self.player.trail.circles[i].0 <= 0.0 {
                self.player.trail.circles.remove(i);
            } else {
                self.player.trail.circles[i].1.radius = self.player.trail.circles[i].0 * (self.player.circle.radius / self.trail_time);
                i += 1;
            }
        }
        //self.player.trail.circles.truncate(100);

        if self.bgtimer <= 0.0 {
            self.bgtimer += 3.0;
            let mut rng = rand::thread_rng();
            let colord = rand::distributions::Range::new(0u8, 30u8);
            let direction = rng.gen::<f64>() > 0.4;
            self.effects.push(Box::new(BackgroundEffect::new((0, 0, 0), (colord.ind_sample(&mut rng), colord.ind_sample(&mut rng), colord.ind_sample(&mut rng)), 200, direction, 10.0, 150)) as Box<Effect>);
        }

        let mut i = 0;
        while i < self.effects.len() {
            match self.effects[i].update(delta) {
                EffectChange::Ended => {
                    self.effects.remove(i);
                }
                EffectChange::None => {
                    i += 1;
                }
            }
        }

        let mut i = 0;
        while i < self.middle_effects.len() {
            match self.middle_effects[i].update(delta) {
                EffectChange::Ended => {
                    self.middle_effects.remove(i);
                }
                EffectChange::None => {
                    i += 1;
                }
            }
        }

        if self.targets.len() == 0 {
            game.switch_scene(ResultState::initial(Some(self.time)));
        }
    }

    fn handle_event(&mut self, game: &mut GameState, event: Event) {
        match event {
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                game.switch_scene(ResultState::initial(None));
            },
            _ => {}
        }
    }

    fn quit_on_esc(&self) -> bool { false }

    fn render(&self, game: &mut GameState, renderer: &mut Renderer, delta: f64) {
        renderer.set_draw_color(Color::RGB(0, 0, 0));
        renderer.clear();

        for effect in &self.effects {
            effect.render(renderer, delta);
        }

        for effect in &self.middle_effects {
            effect.render(renderer, delta);
        }

        let mouse_p = Pointf64::new(game.inputs.mouse_x as f64, game.inputs.mouse_y as f64);
        let player_p = self.player.circle.point;
        let player_ps = self.player.circle.point.as_sdl();

        // End coordinate of the targeting line
        let target = player_p + (Pointf64::of_rad(self.aim.angle) * self.aim.length);
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(player_ps, target.as_sdl() ).unwrap();

        if self.player.circle.velocity.length() < STOP_LIMIT {
            renderer.set_draw_color(Color::RGB(50, 50, 255));
        } else {
            renderer.set_draw_color(Color::RGB(50, 255, 50));
        }
        draw_circle(renderer, player_ps, self.player.circle.radius as i32);

        renderer.set_draw_color(Color::RGB(50, 50, 255));
        for &(_age, circle) in self.player.trail.circles.iter() {
            draw_circle(renderer, circle.point.as_sdl(), circle.radius as i32);
        }

        for circle in self.targets.iter() {
            let ir = circle.radius as i32;
            let ur = ir as u8;
            if ur > 23 {
                renderer.set_draw_color(Color::RGB(150, 255, 255));
            } else {
                renderer.set_draw_color(Color::RGB(150, ur * 12 - 32, ur * 12 - 32));
            }
            draw_circle(renderer, circle.point.as_sdl(), ir);
        }

        for wall in self.walls.iter() {
            renderer.set_draw_color(Color::RGB(255, 255, 255));
            wall.draw(renderer);
        }

        draw_text(renderer, &game.info_font,
                  10, 10,
                  Color::RGBA(255, 255, 255, 255),
                  &("FPS: ".to_owned() + &(game.fps as i32).to_string()));
        draw_text(renderer, &game.info_font,
                  10, 30,
                  Color::RGBA(255, 255, 255, 255),
                  &("Time: ".to_owned() + &(self.time as i32).to_string() + "s"));
    }
}

struct ResultState {
    time: Option<f64>,
    timehere: f64,
    effects: Vec<Box<Effect>>,
    bgtimer: f64
}

impl ResultState {
    fn initial(time: Option<f64>) -> Self {
        ResultState {
            time: time,
            timehere: 0.0,
            effects: vec![],
            bgtimer: 0.0
        }
    }
}

impl SceneObject for ResultState {
    fn update(&mut self, game: &mut GameState, delta: f64) {
        self.timehere += delta;

        self.bgtimer -= delta;

        let clicked = !game.inputs.last_mouse.left() && game.inputs.mouse.left();
        let mx = game.inputs.mouse_x;
        let my = game.inputs.mouse_y;

        if clicked && self.timehere > 2.0 {
            if mx > 145 && mx < 255 && my > 380 && my < 420 { 
                game.switch_scene(PlayState::initial(10, 5.0, 0.8)); // Easier
            } else if mx > 320 && mx < 480 && my > 380 && my < 420 { 
                game.switch_scene(PlayState::initial(15, 3.5, 0.8)); // Medium
            } else if mx > 545 && mx < 655 && my > 380 && my < 420 { 
                game.switch_scene(PlayState::initial(20, 2.0, 0.6)); // Hard
            } else if mx > 145 && mx < 255 && my > 480 && my < 520 { 
                game.switch_scene(PlayState::initial(10, 5.0, 0.6)); // Easy
            } else if mx > 320 && mx < 480 && my > 480 && my < 520 { 
                game.switch_scene(PlayState::initial(15, 3.0, 0.6)); // Mediumer
            } else if mx > 545 && mx < 655 && my > 480 && my < 520 { 
                game.switch_scene(PlayState::initial(20, 1.5, 0.4)); // Harder
            } 
        }

        if self.bgtimer <= 0.0 {
            self.bgtimer += 2.5;
            let mut rng = rand::thread_rng();
            let colord = rand::distributions::Range::new(10u8, 70u8);
            let direction = rng.gen::<f64>() > 0.5;
            self.effects.push(Box::new(BackgroundEffect::new((0, 0, 0), (colord.ind_sample(&mut rng), colord.ind_sample(&mut rng), colord.ind_sample(&mut rng)), 200, direction, 10.0, 150)) as Box<Effect>);
        }

        let mut i = 0;
        while i < self.effects.len() {
            match self.effects[i].update(delta) {
                EffectChange::Ended => {
                    self.effects.remove(i);
                }
                EffectChange::None => {
                    i += 1;
                }
            }
        }
    }

    fn handle_event(&mut self, game: &mut GameState, event: Event) {
        match event {
            Event::KeyDown { keycode: Some(Keycode::Space), .. } => {
                game.switch_scene(PlayState::initial(100, 0.3, 2.0));
            }
            _ => {}
        }
    }

    fn render(&self, game: &mut GameState, renderer: &mut Renderer, delta: f64) {
        renderer.set_draw_color(Color::RGB(0, 0, 0));
        renderer.clear();

        for effect in &self.effects {
            effect.render(renderer, delta);
        }

        if let Some(time) = self.time {
            draw_text_centered(renderer, &game.score_font, 400, 200, Color::RGB(255, 255, 255),
                               &("Time taken: ".to_owned() + &(time as i32).to_string() + "s"));
        } else {
            draw_text_centered(renderer, &game.score_font, 400, 200, Color::RGB(255, 255, 255),
                               "Circlin'");
        }

        draw_text(renderer, &game.info_font, 20, 10, Color::RGB(200, 200, 200),
                  "Controls:");
        draw_text(renderer, &game.info_font, 20, 30, Color::RGB(200, 200, 200),
                  "Hold left mouse button - apply force");
        draw_text(renderer, &game.info_font, 20, 50, Color::RGB(200, 200, 200),
                  "ESC - give up");

        draw_text(renderer, &game.info_font, 20, 100, Color::RGB(200, 200, 200),
                  "Catch 'em all to win.");
        draw_text(renderer, &game.info_font, 20, 120, Color::RGB(200, 200, 200),
                  "Hint: Your tail can capture targets too.");
        
        draw_text_centered(renderer, &game.text_font, 400, 300, Color::RGB(255, 255, 255),
                           "Select a difficulty:");
        
        draw_text_centered(renderer, &game.text_font, 200, 400, Color::RGB(200, 200, 200),
                           "Easier");
        draw_text(renderer, &game.info_font, 145, 425, Color::RGB(200, 200, 200),
                  "Spawn time: 5s");
        draw_text(renderer, &game.info_font, 145, 445, Color::RGB(200, 200, 200),
                  "Longer tail");
        
        // Easier
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(145, 380), Point::new(255, 380)).unwrap();
        renderer.draw_line(Point::new(145, 380), Point::new(145, 420)).unwrap();
        renderer.draw_line(Point::new(255, 380), Point::new(255, 420)).unwrap();
        renderer.draw_line(Point::new(145, 420), Point::new(255, 420)).unwrap();

        draw_text_centered(renderer, &game.text_font, 200, 500, Color::RGB(200, 200, 200),
                           "Easy");
        draw_text(renderer, &game.info_font, 145, 525, Color::RGB(200, 200, 200),
                  "Spawn time: 5s");
        
        // Easy
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(145, 480), Point::new(255, 480)).unwrap();
        renderer.draw_line(Point::new(145, 480), Point::new(145, 520)).unwrap();
        renderer.draw_line(Point::new(255, 480), Point::new(255, 520)).unwrap();
        renderer.draw_line(Point::new(145, 520), Point::new(255, 520)).unwrap();

        draw_text_centered(renderer, &game.text_font, 400, 400, Color::RGB(200, 200, 200),
                           "Medium");
        draw_text(renderer, &game.info_font, 320, 425, Color::RGB(200, 200, 200),
                  "Spawn time: 3.5s");
        draw_text(renderer, &game.info_font, 320, 445, Color::RGB(200, 200, 200),
                  "Longer tail");

        // Medium
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(320, 380), Point::new(480, 380)).unwrap();
        renderer.draw_line(Point::new(320, 380), Point::new(320, 420)).unwrap();
        renderer.draw_line(Point::new(480, 380), Point::new(480, 420)).unwrap();
        renderer.draw_line(Point::new(320, 420), Point::new(480, 420)).unwrap();

        draw_text_centered(renderer, &game.text_font, 400, 500, Color::RGB(200, 200, 200),
                           "Mediumer");
        draw_text(renderer, &game.info_font, 320, 525, Color::RGB(200, 200, 200),
                  "Spawn time: 3s");

        // Medium
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(320, 480), Point::new(480, 480)).unwrap();
        renderer.draw_line(Point::new(320, 480), Point::new(320, 520)).unwrap();
        renderer.draw_line(Point::new(480, 480), Point::new(480, 520)).unwrap();
        renderer.draw_line(Point::new(320, 520), Point::new(480, 520)).unwrap();

        draw_text_centered(renderer, &game.text_font, 600, 400, Color::RGB(200, 200, 200),
                           "Hard");
        draw_text(renderer, &game.info_font, 545, 425, Color::RGB(200, 200, 200),
                  "Spawn time: 2s");

        // Hard
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(545, 380), Point::new(655, 380)).unwrap();
        renderer.draw_line(Point::new(545, 380), Point::new(545, 420)).unwrap();
        renderer.draw_line(Point::new(655, 380), Point::new(655, 420)).unwrap();
        renderer.draw_line(Point::new(545, 420), Point::new(655, 420)).unwrap();

        draw_text_centered(renderer, &game.text_font, 600, 500, Color::RGB(200, 200, 200),
                           "Harder");
        draw_text(renderer, &game.info_font, 545, 525, Color::RGB(200, 200, 200),
                  "Spawn time: 1.5s");
        draw_text(renderer, &game.info_font, 545, 545, Color::RGB(200, 200, 200),
                  "Short tail");
        draw_text(renderer, &game.info_font, 545, 565, Color::RGB(200, 200, 200),
                  "The true test");
        
        // Harder
        renderer.set_draw_color(Color::RGB(255, 255, 255));
        renderer.draw_line(Point::new(545, 480), Point::new(655, 480)).unwrap();
        renderer.draw_line(Point::new(545, 480), Point::new(545, 520)).unwrap();
        renderer.draw_line(Point::new(655, 480), Point::new(655, 520)).unwrap();
        renderer.draw_line(Point::new(545, 520), Point::new(655, 520)).unwrap();
    }
}

fn draw_circle(renderer: &mut Renderer, center: Point, radius: i32) {
    let r2 = radius*radius;

    for x in 0..radius {
        let y = sqrt(r2 - x*x);
        renderer.draw_line(Point::new(center.x() - x, center.y() - y),
                           Point::new(center.x() - x, center.y() + y)).unwrap();
        renderer.draw_line(Point::new(center.x() + x, center.y() - y),
                           Point::new(center.x() + x, center.y() + y)).unwrap();
    } 
}

fn draw_text(renderer: &mut Renderer, font: &Font, x: i32, y: i32, color: Color, text: &str) {
    let surface = font.render(text).blended(color).unwrap();
    let mut texture = renderer.create_texture_from_surface(&surface).unwrap();
    let TextureQuery { width, height , ..} = texture.query();
    renderer.copy(&mut texture, None, Some(Rect::new(x, y, width, height))).unwrap();
}

fn draw_text_centered(renderer: &mut Renderer, font: &Font, x: i32, y: i32, color: Color, text: &str) {
    let surface = font.render(text).blended(color).unwrap();
    let mut texture = renderer.create_texture_from_surface(&surface).unwrap();
    let TextureQuery { width, height , ..} = texture.query();
    renderer.copy(&mut texture, None, Some(Rect::new(x - (width/2) as i32, y - (height/2) as i32, width, height))).unwrap();
}

fn run(sdl_context: sdl2::Sdl, ttf_context: sdl2_ttf::Sdl2TtfContext, mut renderer: Renderer) {
    let mouse = sdl_context.mouse();

    let mut timer = sdl_context.timer().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_time = clock_ticks::precise_time_s();

    let mut game = GameState { 
        changes: vec![],
        inputs: Inputs::new(),
        fps: 0.0,
        info_font: ttf_context.load_font(std::path::Path::new("Roboto-Medium.ttf"), 16).unwrap(),
        score_font: ttf_context.load_font(std::path::Path::new("Roboto-Medium.ttf"), 64).unwrap(),
        text_font: ttf_context.load_font(std::path::Path::new("Roboto-Medium.ttf"), 32).unwrap()
    };
    // let mut scene = Scene::Play(PlayState::initial(10));
    let mut scene = Box::new(ResultState::initial(None)) as Box<SceneObject>;

    let mut framecount = 0;
    let mut fpstime = 0.0;
    'running: loop {
        let now = clock_ticks::precise_time_s();
        let delta = now - last_time;
        last_time = now;
        fpstime += delta;
        if fpstime >= 1.0 {
            game.fps = (framecount as f64) * fpstime;
            fpstime -= 1.0;
            framecount = 0;
        }
        framecount += 1;

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} => {
                    break 'running;
                },
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    if scene.quit_on_esc() {
                        break 'running;
                    } else {
                        scene.handle_event(&mut game, event);
                    }
                },
                _ => {
                    scene.handle_event(&mut game, event);
                }
            }
        }

        let (mb, mx, my) = mouse.mouse_state();
        game.inputs.last_mouse = game.inputs.mouse;
        game.inputs.mouse = mb;
        game.inputs.mouse_x = mx;
        game.inputs.mouse_y = my;

        // DRAW
        scene.update(&mut game, delta);
        scene.render(&mut game, &mut renderer, delta);

        for change in game.changes.drain(..) {
            match change {
                GameStateChange::SwitchScene(new_scene) => {
                    scene = new_scene;
                }
            }
        }

        renderer.present();

        let now = clock_ticks::precise_time_s();
        let delta = now - last_time;
        if (16.66 - delta * 1000.0) > 1.0 {
            timer.delay((16.66 - delta * 1000.0) as u32 - 1);
        }
    } 
}

fn main() {
    println!("linked sdl2_ttf: {}", sdl2_ttf::get_linked_version());
    let sdl_context = sdl2::init().unwrap();
    let ttf_context = sdl2_ttf::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("Circlin'", 800, 600)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut renderer = window.renderer().build().unwrap();
    renderer.set_blend_mode(BlendMode::Blend);
    run(sdl_context, ttf_context, renderer);
}
