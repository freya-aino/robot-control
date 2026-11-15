//! The composition arcs beyond what the G1 layers exercise: inherits and
//! specializes, list edits (delete, reorder, explicit), layer offsets on
//! sublayers and reference arcs, relationship-target remapping through a
//! reference, and standalone target specs.
//!
//! Layers are authored programmatically ([`Layer`]/[`Spec`]), since the
//! vendored robot files do not use these constructs — which is exactly why
//! they were previously reported as unsupported rather than implemented.

use robot_usd::error::UsdError;
use robot_usd::layer::{Layer, Spec, SpecKind};
use robot_usd::stage::Stage;
use robot_usd::value::{Arc, ListOp, Specifier, Value};

fn tokens(items: &[&str]) -> Value {
    Value::Array(
        items
            .iter()
            .map(|s| Value::Token((*s).to_owned()))
            .collect(),
    )
}

fn pseudo_root(fields: Vec<(String, Value)>) -> Spec {
    Spec {
        path: "/".to_owned(),
        kind: SpecKind::PseudoRoot,
        fields,
    }
}

fn prim(path: &str, mut fields: Vec<(String, Value)>) -> Spec {
    fields.insert(
        0,
        ("specifier".to_owned(), Value::Specifier(Specifier::Def)),
    );
    Spec {
        path: path.to_owned(),
        kind: SpecKind::Prim,
        fields,
    }
}

fn attr(path: &str, value: Value) -> Spec {
    Spec {
        path: path.to_owned(),
        kind: SpecKind::Attribute,
        fields: vec![
            ("typeName".to_owned(), Value::Token("double".to_owned())),
            ("default".to_owned(), value),
        ],
    }
}

fn list(f: impl FnOnce(&mut ListOp)) -> Value {
    let mut op = ListOp::default();
    f(&mut op);
    Value::ListOp(Box::new(op))
}

fn paths_list(f: impl FnOnce(&mut ListOp)) -> (String, Value) {
    ("inherits".to_owned(), list(f))
}

#[test]
fn an_inherited_class_contributes_and_local_opinions_win() {
    let layer = Layer {
        identifier: "root".to_owned(),
        specs: vec![
            pseudo_root(vec![(
                "primChildren".to_owned(),
                tokens(&["_class_arm", "arm"]),
            )]),
            prim(
                "/_class_arm",
                vec![
                    ("specifier".to_owned(), Value::Specifier(Specifier::Class)),
                    ("properties".to_owned(), tokens(&["reach", "payload"])),
                ],
            ),
            attr("/_class_arm.reach", Value::Double(0.8)),
            attr("/_class_arm.payload", Value::Double(3.0)),
            prim(
                "/arm",
                vec![
                    {
                        let (k, v) = paths_list(|op| {
                            op.appended.push(Value::Path("/_class_arm".to_owned()));
                        });
                        (k, v)
                    },
                    ("properties".to_owned(), tokens(&["payload"])),
                ],
            ),
            attr("/arm.payload", Value::Double(5.0)),
        ],
    };
    let stage = Stage::from_layers(vec![layer]).unwrap();
    let arm = stage.prim("/arm").unwrap();
    // Inherited from the class:
    assert_eq!(
        arm.property("reach").unwrap().default,
        Some(Value::Double(0.8))
    );
    // The local opinion beats the inherited one:
    assert_eq!(
        arm.property("payload").unwrap().default,
        Some(Value::Double(5.0))
    );
}

#[test]
fn specializes_is_weaker_than_a_reference() {
    let layer = Layer {
        identifier: "root".to_owned(),
        specs: vec![
            pseudo_root(vec![(
                "primChildren".to_owned(),
                tokens(&["_base", "_ref", "robot"]),
            )]),
            prim("/_base", vec![("properties".to_owned(), tokens(&["gain"]))]),
            attr("/_base.gain", Value::Double(1.0)),
            prim("/_ref", vec![("properties".to_owned(), tokens(&["gain"]))]),
            attr("/_ref.gain", Value::Double(2.0)),
            prim(
                "/robot",
                vec![
                    (
                        "specializes".to_owned(),
                        list(|op| op.appended.push(Value::Path("/_base".to_owned()))),
                    ),
                    (
                        "references".to_owned(),
                        list(|op| {
                            op.appended.push(Value::Reference(Box::new(Arc {
                                asset_path: String::new(),
                                prim_path: "/_ref".to_owned(),
                                layer_offset: (0.0, 1.0),
                                custom_data: Vec::new(),
                            })));
                        }),
                    ),
                ],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![layer]).unwrap();
    let robot = stage.prim("/robot").unwrap();
    assert_eq!(
        robot.property("gain").unwrap().default,
        Some(Value::Double(2.0)),
        "the reference must beat the specializes arc"
    );
}

#[test]
fn list_edits_delete_reorder_and_replace_across_layers() {
    let strong = Layer {
        identifier: "strong".to_owned(),
        specs: vec![
            pseudo_root(vec![
                ("subLayers".to_owned(), tokens(&["weak"])),
                ("primChildren".to_owned(), tokens(&["p"])),
            ]),
            prim(
                "/p",
                vec![(
                    "apiSchemas".to_owned(),
                    list(|op| {
                        op.deleted.push(Value::Token("B".to_owned()));
                        op.ordered.push(Value::Token("C".to_owned()));
                        op.ordered.push(Value::Token("A".to_owned()));
                    }),
                )],
            ),
        ],
    };
    let weak = Layer {
        identifier: "weak".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["p"]))]),
            prim(
                "/p",
                vec![(
                    "apiSchemas".to_owned(),
                    list(|op| {
                        for t in ["A", "B", "C"] {
                            op.appended.push(Value::Token(t.to_owned()));
                        }
                    }),
                )],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![strong, weak]).unwrap();
    assert_eq!(
        stage.prim("/p").unwrap().api_schemas,
        vec!["C".to_owned(), "A".to_owned()],
        "B deleted, then C and A reordered"
    );

    // An explicit list in the strong layer replaces the weaker list outright.
    let explicit = Layer {
        identifier: "explicit".to_owned(),
        specs: vec![
            pseudo_root(vec![
                ("subLayers".to_owned(), tokens(&["weak2"])),
                ("primChildren".to_owned(), tokens(&["p"])),
            ]),
            prim(
                "/p",
                vec![(
                    "apiSchemas".to_owned(),
                    list(|op| {
                        op.is_explicit = true;
                        op.explicit.push(Value::Token("Only".to_owned()));
                    }),
                )],
            ),
        ],
    };
    let weak2 = Layer {
        identifier: "weak2".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["p"]))]),
            prim(
                "/p",
                vec![(
                    "apiSchemas".to_owned(),
                    list(|op| op.appended.push(Value::Token("Gone".to_owned()))),
                )],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![explicit, weak2]).unwrap();
    assert_eq!(
        stage.prim("/p").unwrap().api_schemas,
        vec!["Only".to_owned()]
    );
}

#[test]
fn sublayer_and_reference_offsets_map_time_samples() {
    // The sublayer's samples at t are shown at 2 + 0.5 t.
    let root = Layer {
        identifier: "root".to_owned(),
        specs: vec![pseudo_root(vec![
            ("subLayers".to_owned(), tokens(&["anim"])),
            (
                "subLayerOffsets".to_owned(),
                Value::Array(vec![Value::LayerOffset(2.0, 0.5)]),
            ),
            ("primChildren".to_owned(), tokens(&["p"])),
        ])],
    };
    let anim = Layer {
        identifier: "anim".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["p"]))]),
            prim("/p", vec![("properties".to_owned(), tokens(&["x"]))]),
            Spec {
                path: "/p.x".to_owned(),
                kind: SpecKind::Attribute,
                fields: vec![
                    ("typeName".to_owned(), Value::Token("double".to_owned())),
                    (
                        "timeSamples".to_owned(),
                        Value::TimeSamples(vec![
                            (0.0, Value::Double(1.0)),
                            (10.0, Value::Double(2.0)),
                        ]),
                    ),
                ],
            },
        ],
    };
    let stage = Stage::from_layers(vec![root, anim]).unwrap();
    let ts = stage
        .prim("/p")
        .unwrap()
        .property("x")
        .unwrap()
        .time_samples
        .clone()
        .unwrap();
    assert_eq!(
        ts,
        vec![(2.0, Value::Double(1.0)), (7.0, Value::Double(2.0))]
    );

    // The same mapping through a reference arc's own offset.
    let root = Layer {
        identifier: "root2".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["q", "clip"]))]),
            prim("/clip", vec![("properties".to_owned(), tokens(&["x"]))]),
            Spec {
                path: "/clip.x".to_owned(),
                kind: SpecKind::Attribute,
                fields: vec![
                    ("typeName".to_owned(), Value::Token("double".to_owned())),
                    (
                        "timeSamples".to_owned(),
                        Value::TimeSamples(vec![(4.0, Value::Double(9.0))]),
                    ),
                ],
            },
            prim(
                "/q",
                vec![(
                    "references".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Reference(Box::new(Arc {
                            asset_path: String::new(),
                            prim_path: "/clip".to_owned(),
                            layer_offset: (1.0, 2.0),
                            custom_data: Vec::new(),
                        })));
                    }),
                )],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![root]).unwrap();
    let ts = stage
        .prim("/q")
        .unwrap()
        .property("x")
        .unwrap()
        .time_samples
        .clone()
        .unwrap();
    assert_eq!(ts, vec![(9.0, Value::Double(9.0))]);
}

#[test]
fn relationship_targets_remap_through_a_reference() {
    // /robot references /tpl; a target authored inside /tpl must surface
    // as the corresponding /robot path, and one outside stays as authored.
    let layer = Layer {
        identifier: "root".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["tpl", "robot"]))]),
            prim(
                "/tpl",
                vec![("primChildren".to_owned(), tokens(&["link", "joint"]))],
            ),
            prim("/tpl/link", vec![]),
            prim(
                "/tpl/joint",
                vec![("properties".to_owned(), tokens(&["physics:body0"]))],
            ),
            Spec {
                path: "/tpl/joint.physics:body0".to_owned(),
                kind: SpecKind::Relationship,
                fields: vec![(
                    "targetPaths".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Path("/tpl/link".to_owned()));
                        op.appended.push(Value::Path("/elsewhere".to_owned()));
                    }),
                )],
            },
            prim(
                "/robot",
                vec![(
                    "references".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Reference(Box::new(Arc {
                            asset_path: String::new(),
                            prim_path: "/tpl".to_owned(),
                            layer_offset: (0.0, 1.0),
                            custom_data: Vec::new(),
                        })));
                    }),
                )],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![layer]).unwrap();
    let joint = stage.prim("/robot/joint").unwrap();
    assert_eq!(
        joint.property("physics:body0").unwrap().targets,
        vec!["/robot/link".to_owned(), "/elsewhere".to_owned()]
    );
}

#[test]
fn standalone_target_specs_are_accepted_only_when_fieldless() {
    let base = |fields: Vec<(String, Value)>| Layer {
        identifier: "root".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["p"]))]),
            prim("/p", vec![("properties".to_owned(), tokens(&["rel"]))]),
            Spec {
                path: "/p.rel".to_owned(),
                kind: SpecKind::Relationship,
                fields: vec![(
                    "targetPaths".to_owned(),
                    list(|op| op.appended.push(Value::Path("/p".to_owned()))),
                )],
            },
            Spec {
                path: "/p.rel[/p]".to_owned(),
                kind: SpecKind::RelationshipTarget,
                fields,
            },
        ],
    };
    let stage = Stage::from_layers(vec![base(Vec::new())]).unwrap();
    assert_eq!(
        stage.prim("/p").unwrap().property("rel").unwrap().targets,
        vec!["/p".to_owned()]
    );
    let err =
        Stage::from_layers(vec![base(vec![("custom".to_owned(), Value::Bool(true))])]).unwrap_err();
    assert!(matches!(err, UsdError::Unsupported(_)), "{err:?}");
}

#[test]
fn a_sibling_target_is_kept_when_the_referenced_prim_shares_the_composed_name() {
    // /arm references /tpl/arm: the shared trailing name must not widen
    // the remap root to /tpl -> /, which would rewrite the sibling target
    // /tpl/base that the module contract promises is kept as authored.
    let layer = Layer {
        identifier: "root".to_owned(),
        specs: vec![
            pseudo_root(vec![("primChildren".to_owned(), tokens(&["tpl", "arm"]))]),
            prim(
                "/tpl",
                vec![("primChildren".to_owned(), tokens(&["arm", "base"]))],
            ),
            prim(
                "/tpl/arm",
                vec![
                    ("primChildren".to_owned(), tokens(&["tip"])),
                    ("properties".to_owned(), tokens(&["out"])),
                ],
            ),
            prim("/tpl/arm/tip", vec![]),
            prim("/tpl/base", vec![]),
            Spec {
                path: "/tpl/arm.out".to_owned(),
                kind: SpecKind::Relationship,
                fields: vec![(
                    "targetPaths".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Path("/tpl/arm/tip".to_owned()));
                        op.appended.push(Value::Path("/tpl/base".to_owned()));
                    }),
                )],
            },
            prim(
                "/arm",
                vec![(
                    "references".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Reference(Box::new(Arc {
                            asset_path: String::new(),
                            prim_path: "/tpl/arm".to_owned(),
                            layer_offset: (0.0, 1.0),
                            custom_data: Vec::new(),
                        })));
                    }),
                )],
            ),
        ],
    };
    let stage = Stage::from_layers(vec![layer]).unwrap();
    let arm = stage.prim("/arm").unwrap();
    assert_eq!(
        arm.property("out").unwrap().targets,
        vec!["/arm/tip".to_owned(), "/tpl/base".to_owned()],
        "inside the arc remapped, the sibling kept as authored"
    );
}

#[test]
fn a_stronger_variant_selection_replaces_a_weaker_one() {
    // /p references /q. /q selects color=blue for itself; /p selects
    // color=red. The stronger selection must *replace* the weaker one:
    // only the red variant's opinions may surface, and blue's must not
    // leak in through the deselected variant.
    let variant = |path: &str, prop: &str, v: f64| {
        vec![
            Spec {
                path: path.to_owned(),
                kind: SpecKind::Variant,
                fields: vec![("properties".to_owned(), tokens(&[prop]))],
            },
            attr(&format!("{path}.{prop}"), Value::Double(v)),
        ]
    };
    let mut specs = vec![
        pseudo_root(vec![("primChildren".to_owned(), tokens(&["q", "p"]))]),
        prim(
            "/q",
            vec![(
                "variantSelection".to_owned(),
                Value::Dictionary(vec![("color".to_owned(), Value::String("blue".to_owned()))]),
            )],
        ),
        prim(
            "/p",
            vec![
                (
                    "references".to_owned(),
                    list(|op| {
                        op.appended.push(Value::Reference(Box::new(Arc {
                            asset_path: String::new(),
                            prim_path: "/q".to_owned(),
                            layer_offset: (0.0, 1.0),
                            custom_data: Vec::new(),
                        })));
                    }),
                ),
                (
                    "variantSelection".to_owned(),
                    Value::Dictionary(vec![("color".to_owned(), Value::String("red".to_owned()))]),
                ),
            ],
        ),
    ];
    specs.extend(variant("/q{color=red}", "warmth", 1.0));
    specs.extend(variant("/q{color=blue}", "coolness", 2.0));
    let layer = Layer {
        identifier: "root".to_owned(),
        specs,
    };
    let stage = Stage::from_layers(vec![layer]).unwrap();
    let p = stage.prim("/p").unwrap();
    assert!(p.property("warmth").is_some(), "the red variant applies");
    assert!(
        p.property("coolness").is_none(),
        "the deselected blue variant must not leak in"
    );
}

#[test]
fn absurd_prim_nesting_is_refused_not_a_stack_overflow() {
    let mut specs = vec![pseudo_root(vec![(
        "primChildren".to_owned(),
        tokens(&["p0"]),
    )])];
    let mut path = String::new();
    for i in 0..300 {
        path.push_str(&format!("/p{i}"));
        let fields = vec![("primChildren".to_owned(), tokens(&[&format!("p{}", i + 1)]))];
        specs.push(prim(&path, fields));
    }
    let layer = Layer {
        identifier: "root".to_owned(),
        specs,
    };
    assert!(matches!(
        Stage::from_layers(vec![layer]),
        Err(UsdError::Unsupported(_))
    ));
}
