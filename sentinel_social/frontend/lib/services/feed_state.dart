import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';

class UserPost {
  final String imagePath;
  final String caption;
  final double toxicityScore;
  final String extractedText;
  final bool isSafe;
  final DateTime timestamp;

  UserPost({
    required this.imagePath,
    required this.caption,
    required this.toxicityScore,
    required this.extractedText,
    required this.isSafe,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
    'imagePath': imagePath,
    'caption': caption,
    'toxicityScore': toxicityScore,
    'extractedText': extractedText,
    'isSafe': isSafe,
    'timestamp': timestamp.toIso8601String(),
  };

  factory UserPost.fromJson(Map<String, dynamic> json) => UserPost(
    imagePath: json['imagePath'] as String,
    caption: json['caption'] as String? ?? '',
    toxicityScore: (json['toxicityScore'] as num).toDouble(),
    extractedText: json['extractedText'] as String? ?? '',
    isSafe: json['isSafe'] as bool,
    timestamp: DateTime.parse(json['timestamp'] as String),
  );
}

class ScanHistoryEntry {
  final String imagePath;
  final double toxicityScore;
  final bool isSafe;
  final DateTime timestamp;

  ScanHistoryEntry({
    required this.imagePath,
    required this.toxicityScore,
    required this.isSafe,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
    'imagePath': imagePath,
    'toxicityScore': toxicityScore,
    'isSafe': isSafe,
    'timestamp': timestamp.toIso8601String(),
  };

  factory ScanHistoryEntry.fromJson(Map<String, dynamic> json) =>
      ScanHistoryEntry(
        imagePath: json['imagePath'] as String,
        toxicityScore: (json['toxicityScore'] as num).toDouble(),
        isSafe: json['isSafe'] as bool,
        timestamp: DateTime.parse(json['timestamp'] as String),
      );
}

class FeedState extends ChangeNotifier {
  static const _postsKey = 'sentinel_user_posts';
  static const _historyKey = 'sentinel_scan_history';
  static const int maxPosts = 3;
  static const int maxHistory = 10;

  List<UserPost> _userPosts = [];
  List<ScanHistoryEntry> _scanHistory = [];

  List<UserPost> get userPosts => List.unmodifiable(_userPosts);
  List<ScanHistoryEntry> get scanHistory => List.unmodifiable(_scanHistory);

  Future<void> loadFromStorage() async {
    final prefs = await SharedPreferences.getInstance();

    final postsJson = prefs.getString(_postsKey);
    if (postsJson != null) {
      final List<dynamic> decoded = json.decode(postsJson);
      _userPosts = decoded
          .map((e) => UserPost.fromJson(e as Map<String, dynamic>))
          .where((p) => File(p.imagePath).existsSync())
          .toList();
    }

    final historyJson = prefs.getString(_historyKey);
    if (historyJson != null) {
      final List<dynamic> decoded = json.decode(historyJson);
      _scanHistory = decoded
          .map((e) => ScanHistoryEntry.fromJson(e as Map<String, dynamic>))
          .toList();
    }

    notifyListeners();
  }

  Future<void> _savePosts() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = json.encode(_userPosts.map((p) => p.toJson()).toList());
    await prefs.setString(_postsKey, encoded);
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = json.encode(_scanHistory.map((h) => h.toJson()).toList());
    await prefs.setString(_historyKey, encoded);
  }

  Future<String> _copyImageToAppDir(String originalPath) async {
    final appDir = await getApplicationDocumentsDirectory();
    final sentinelDir = Directory('${appDir.path}/sentinel_posts');
    if (!sentinelDir.existsSync()) {
      sentinelDir.createSync(recursive: true);
    }
    final fileName = 'post_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final newPath = '${sentinelDir.path}/$fileName';
    await File(originalPath).copy(newPath);
    return newPath;
  }

  Future<void> addPost({
    required String imagePath,
    required double toxicityScore,
    required String extractedText,
    required bool isSafe,
    String caption = '',
  }) async {
    final savedPath = await _copyImageToAppDir(imagePath);

    final post = UserPost(
      imagePath: savedPath,
      caption: caption,
      toxicityScore: toxicityScore,
      extractedText: extractedText,
      isSafe: isSafe,
      timestamp: DateTime.now(),
    );

    _userPosts.insert(0, post);

    while (_userPosts.length > maxPosts) {
      final removed = _userPosts.removeLast();
      final file = File(removed.imagePath);
      if (file.existsSync()) {
        file.deleteSync();
      }
    }

    await _savePosts();
    notifyListeners();
  }

  Future<void> addScanHistory({
    required String imagePath,
    required double toxicityScore,
    required bool isSafe,
  }) async {
    _scanHistory.insert(
      0,
      ScanHistoryEntry(
        imagePath: imagePath,
        toxicityScore: toxicityScore,
        isSafe: isSafe,
        timestamp: DateTime.now(),
      ),
    );

    while (_scanHistory.length > maxHistory) {
      _scanHistory.removeLast();
    }

    await _saveHistory();
    notifyListeners();
  }
}

class FeedStateProvider extends InheritedNotifier<FeedState> {
  const FeedStateProvider({
    super.key,
    required FeedState feedState,
    required super.child,
  }) : super(notifier: feedState);

  static FeedState of(BuildContext context) {
    final provider =
        context.dependOnInheritedWidgetOfExactType<FeedStateProvider>();
    return provider!.notifier!;
  }
}
