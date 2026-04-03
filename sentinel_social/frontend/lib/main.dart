import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/feed_screen.dart';
import 'services/feed_state.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final feedState = FeedState();
  await feedState.loadFromStorage();
  runApp(SentinelApp(feedState: feedState));
}

class SentinelApp extends StatelessWidget {
  final FeedState feedState;
  const SentinelApp({super.key, required this.feedState});

  @override
  Widget build(BuildContext context) {
    return FeedStateProvider(
      feedState: feedState,
      child: MaterialApp(
        title: 'Sentinel-X',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xFF09090B),
          primaryColor: const Color(0xFFC9A84C),
          colorScheme: const ColorScheme.dark(
            primary: Color(0xFFC9A84C),
            secondary: Color(0xFFF0D78C),
            surface: Color(0xFF141416),
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
      ),
    );
  }
}
