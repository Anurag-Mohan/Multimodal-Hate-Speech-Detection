import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/feed_screen.dart';

void main() {
  runApp(const SentinelApp());
}

class SentinelApp extends StatelessWidget {
  const SentinelApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sentinel-X',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0e1117),
        primaryColor: const Color(0xFF00f2fe),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00f2fe),
          secondary: Color(0xFF4facfe),
          surface: Color(0xFF1a1c24),
        ),
        textTheme: GoogleFonts.interTextTheme(
          ThemeData.dark().textTheme,
        ).copyWith(
          displayLarge: GoogleFonts.outfit(fontWeight: FontWeight.bold, color: Colors.white),
          titleLarge: GoogleFonts.outfit(fontWeight: FontWeight.w600, color: Colors.white),
        ),
        useMaterial3: true,
      ),
      home: const FeedScreen(),
    );
  }
}
